import numpy as np


BITS_PER_GROUP = 8


def read_words(name):
    print(f"\nPaste {name}. Empty line to finish:")

    lines = []

    while True:
        line = input("> ").strip()

        if not line:
            break

        lines.append(line)

    if not lines:
        return np.array([], dtype=np.uint32)

    return np.array(
        [int(value) for value in " ".join(lines).split()],
        dtype=np.uint32,
    )


def words_from_bl(bl):
    """
    One sign bit plus BL pulse bits, packed into uint32 words.
    """
    return (bl + 32) // 32


def get_train(
    train,
    feature_idx,
    batch_idx,
    feature_count,
    words_per_train,
    batch_size,
    out_trans,
):
    """
    Reconstruct one packed train from the flattened backend output.
    """
    words = []

    for word_idx in range(words_per_train):
        if out_trans:
            batch_aligned = batch_idx + batch_size * feature_idx

            index = (
                (batch_aligned // feature_count)
                * words_per_train
                * feature_count
                + batch_aligned % feature_count
                + word_idx * feature_count
            )
        else:
            index = (
                feature_idx
                + feature_count * word_idx
                + batch_idx * words_per_train * feature_count
            )

        words.append(train[index])

    return np.asarray(words, dtype=np.uint32)


def unpack_train(words, bl):
    """
    Return:

        sign:
            Bit 0 of the first uint32 word.

        pulses:
            The BL pulse bits in chronological order:
            pulse 0, pulse 1, ..., pulse BL - 1.

    The sign bit is not included in pulses.
    """
    sign = int(words[0]) & 1
    pulses = np.empty(bl, dtype=np.uint8)

    for pulse_idx in range(bl):
        if pulse_idx < 31:
            word_idx = 0
            bit_idx = pulse_idx + 1
        else:
            shifted_idx = pulse_idx - 31
            word_idx = 1 + shifted_idx // 32
            bit_idx = shifted_idx % 32

        pulses[pulse_idx] = (
            int(words[word_idx]) >> bit_idx
        ) & 1

    return sign, pulses


def check_left_alignment(pulses):
    """
    A left-aligned train has the form:

        111111000000

    It must never contain a 0 -> 1 transition.

    All-zero and all-one trains are also left-aligned.
    """
    for idx in range(len(pulses) - 1):
        if pulses[idx] == 0 and pulses[idx + 1] == 1:
            return False, idx

    return True, None


def format_bits(pulses):
    bit_string = "".join(str(int(bit)) for bit in pulses)

    return " ".join(
        bit_string[start:start + BITS_PER_GROUP]
        for start in range(0, len(bit_string), BITS_PER_GROUP)
    )


def inspect_train(
    name,
    feature_idx,
    batch_idx,
    packed_words,
    bl,
):
    sign, pulses = unpack_train(packed_words, bl)
    aligned, violation_idx = check_left_alignment(pulses)

    status = "OK" if aligned else "FAIL"

    print(
        f"B{batch_idx:<3} "
        f"{name}{feature_idx:<3} "
        f"sign={sign} "
        f"ones={int(pulses.sum()):>3}/{bl:<3} "
        f"{status:<4} "
        f"{format_bits(pulses)}"
    )

    if not aligned:
        print(
            f"      first invalid 0 -> 1 transition: "
            f"pulse {violation_idx} -> {violation_idx + 1}"
        )

    return aligned


# ============================================================
# Input
# ============================================================

d_train = read_words("d_train")
x_train = read_words("x_train")

B = int(input("\nBatch size / m_batch: "))
I = int(input("Number of inputs / x_size: "))
O = int(input("Number of outputs / d_size: "))
BL = int(input("BL / current_BL: "))
out_trans = bool(int(input("out_trans, 0 or 1: ")))

W = words_from_bl(BL)


# ============================================================
# Validation
# ============================================================

expected_x_length = I * B * W
expected_d_length = O * B * W

if len(x_train) != expected_x_length:
    raise ValueError(
        f"x_train has {len(x_train)} values, "
        f"expected {expected_x_length}"
    )

if len(d_train) != expected_d_length:
    raise ValueError(
        f"d_train has {len(d_train)} values, "
        f"expected {expected_d_length}"
    )


# ============================================================
# Inspect trains
# ============================================================

x_failures = []
d_failures = []

print("\nExpected order: 111...000")
print("Bits are printed from pulse 0 to pulse BL-1.\n")


for batch_idx in range(B):
    print(f"--- Batch {batch_idx} ---")

    for input_idx in range(I):
        words = get_train(
            train=x_train,
            feature_idx=input_idx,
            batch_idx=batch_idx,
            feature_count=I,
            words_per_train=W,
            batch_size=B,
            out_trans=out_trans,
        )

        aligned = inspect_train(
            name="X",
            feature_idx=input_idx,
            batch_idx=batch_idx,
            packed_words=words,
            bl=BL,
        )

        if not aligned:
            x_failures.append((batch_idx, input_idx))

    for output_idx in range(O):
        words = get_train(
            train=d_train,
            feature_idx=output_idx,
            batch_idx=batch_idx,
            feature_count=O,
            words_per_train=W,
            batch_size=B,
            out_trans=out_trans,
        )

        aligned = inspect_train(
            name="D",
            feature_idx=output_idx,
            batch_idx=batch_idx,
            packed_words=words,
            bl=BL,
        )

        if not aligned:
            d_failures.append((batch_idx, output_idx))

    print()


# ============================================================
# Summary
# ============================================================

total_x = B * I
total_d = B * O

print("Summary")
print(f"X: {total_x - len(x_failures)}/{total_x} left-aligned")
print(f"D: {total_d - len(d_failures)}/{total_d} left-aligned")

if x_failures:
    print(
        "Failed X: "
        + ", ".join(
            f"B{batch_idx}/X{input_idx}"
            for batch_idx, input_idx in x_failures
        )
    )

if d_failures:
    print(
        "Failed D: "
        + ", ".join(
            f"B{batch_idx}/D{output_idx}"
            for batch_idx, output_idx in d_failures
        )
    )