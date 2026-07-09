from pathlib import Path

import numpy as np


BITS_PER_GROUP = 8

DEFAULT_DUMP_FILE = (
    Path(__file__).resolve().parent / "05_repeated_group_train_dump.txt"
)


def read_metadata(lines):
    metadata = {}

    for line in lines:
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        if line.startswith("["):
            break

        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key.strip()] = value.strip()

    return metadata


def read_uint32_section(lines, section_name):
    start_tag = f"[{section_name}]"
    end_tag = f"[/{section_name}]"

    inside = False
    values = []

    for line in lines:
        line = line.strip()

        if line == start_tag:
            inside = True
            continue

        if line == end_tag:
            break

        if inside and line:
            values.extend(int(value) for value in line.split())

    if not values:
        raise ValueError(f"Section [{section_name}] is empty or missing.")

    return np.asarray(values, dtype=np.uint32)


def load_train_dump(path=DEFAULT_DUMP_FILE):
    path = path.expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Dump file not found: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    metadata = read_metadata(lines)

    d_train = read_uint32_section(lines, "d_train")
    x_train = read_uint32_section(lines, "x_train")

    return {
        "d_train": d_train,
        "x_train": x_train,
        "B": int(metadata["batch_size"]),
        "I": int(metadata["input_count"]),
        "O": int(metadata["output_count"]),
        "BL": int(metadata["bl"]),
        "out_trans": bool(int(metadata["out_trans"])),
    }


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
    Return sign and BL pulse bits in chronological order.
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

        pulses[pulse_idx] = (int(words[word_idx]) >> bit_idx) & 1

    return sign, pulses


def check_left_alignment(pulses):
    """
    A left-aligned train has the form 111111000000.
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
# Load dump automatically
# ============================================================

dump = load_train_dump()

d_train = dump["d_train"]
x_train = dump["x_train"]

B = dump["B"]
I = dump["I"]
O = dump["O"]
BL = dump["BL"]
out_trans = dump["out_trans"]

W = words_from_bl(BL)


# ============================================================
# Validation
# ============================================================

expected_x_length = I * B * W
expected_d_length = O * B * W

if len(x_train) != expected_x_length:
    raise ValueError(
        f"x_train has {len(x_train)} values, expected {expected_x_length}"
    )

if len(d_train) != expected_d_length:
    raise ValueError(
        f"d_train has {len(d_train)} values, expected {expected_d_length}"
    )


# ============================================================
# Inspect trains
# ============================================================

x_failures = []
d_failures = []

print(f"Loaded dump: {DEFAULT_DUMP_FILE}")
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