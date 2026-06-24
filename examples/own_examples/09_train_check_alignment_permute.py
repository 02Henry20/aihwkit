from collections import Counter

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

    return np.array(
        [int(value) for value in " ".join(lines).split()],
        dtype=np.uint32,
    )


def words_from_bl(bl):
    """Number of uint32 words for one sign bit and BL pulse bits."""
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
    """Reconstruct one packed train from the flattened backend output."""
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
    Extract the sign and the BL pulse bits.

    Pulse order:
        pulse 0, pulse 1, ..., pulse BL - 1

    The sign bit is excluded from the pulse array.
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


def reconstruct_batch(
    train,
    batch_idx,
    feature_count,
    words_per_train,
    batch_size,
    bl,
    out_trans,
):
    """
    Return one batch as:

        signs:  shape (feature_count,)
        pulses: shape (feature_count, BL)

    Rows are ordered by input/output feature index.
    Columns are pulse positions.
    """
    signs = np.empty(feature_count, dtype=np.uint8)
    pulses = np.empty((feature_count, bl), dtype=np.uint8)

    for feature_idx in range(feature_count):
        words = get_train(
            train=train,
            feature_idx=feature_idx,
            batch_idx=batch_idx,
            feature_count=feature_count,
            words_per_train=words_per_train,
            batch_size=batch_size,
            out_trans=out_trans,
        )

        sign, feature_pulses = unpack_train(words, bl)

        signs[feature_idx] = sign
        pulses[feature_idx] = feature_pulses

    return signs, pulses


def create_left_aligned_reference(pulses):
    """
    Create the left-aligned version of every train.

    Example:
        10100100 -> 11100000
    """
    aligned = np.zeros_like(pulses)

    for feature_idx in range(pulses.shape[0]):
        pulse_count = int(pulses[feature_idx].sum())
        aligned[feature_idx, :pulse_count] = 1

    return aligned
                            

def get_column_counter(pulses):
    """
    Count complete pulse columns across the IN/OUT dimension.

    Each column is represented as a tuple such as:

        (1, 0, 1, 1)

    Counter is used so duplicate columns are handled correctly.
    """
    return Counter(
        tuple(int(bit) for bit in pulses[:, column_idx])
        for column_idx in range(pulses.shape[1])
    )


def analyze_permutation(pulses):
    """
    Check whether pulses are:

        ALIGNED ONLY:
            Already equal to the left-aligned reference.

        PERMUTED OK:
            Not aligned, but all complete columns are a permutation
            of the columns in the left-aligned reference.

        INVALID:
            Column contents or multiplicities do not match.
    """
    aligned_reference = create_left_aligned_reference(pulses)

    actual_columns = get_column_counter(pulses)
    reference_columns = get_column_counter(aligned_reference)

    matched_columns = sum(
        (actual_columns & reference_columns).values()
    )

    if np.array_equal(pulses, aligned_reference):
        status = "ALIGNED ONLY"
    elif actual_columns == reference_columns:
        status = "PERMUTED OK"
    else:
        status = "INVALID"

    return status, matched_columns, aligned_reference


def format_bits(pulses):
    bit_string = "".join(str(int(bit)) for bit in pulses)

    return " ".join(
        bit_string[start:start + BITS_PER_GROUP]
        for start in range(0, len(bit_string), BITS_PER_GROUP)
    )


def print_matrix(name, signs, pulses):
    for feature_idx in range(pulses.shape[0]):
        print(
            f"  {name}{feature_idx:<3} "
            f"s={int(signs[feature_idx])} "
            f"{format_bits(pulses[feature_idx])}"
        )


def inspect_batch_matrix(name, signs, pulses):
    status, matched_columns, _ = analyze_permutation(pulses)

    print(
        f"{name}: {status} "
        f"({matched_columns}/{pulses.shape[1]} columns match)"
    )

    print_matrix(name, signs, pulses)

    return status


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
# Analysis
# ============================================================

summary = {
    "X": Counter(),
    "D": Counter(),
}

print("\nPulse order: pulse 0 -> pulse BL-1")
print("Signs are displayed separately and excluded from the check.\n")


for batch_idx in range(B):
    print(f"--- Batch {batch_idx} ---")

    x_signs, x_pulses = reconstruct_batch(
        train=x_train,
        batch_idx=batch_idx,
        feature_count=I,
        words_per_train=W,
        batch_size=B,
        bl=BL,
        out_trans=out_trans,
    )

    d_signs, d_pulses = reconstruct_batch(
        train=d_train,
        batch_idx=batch_idx,
        feature_count=O,
        words_per_train=W,
        batch_size=B,
        bl=BL,
        out_trans=out_trans,
    )

    x_status = inspect_batch_matrix(
        name="X",
        signs=x_signs,
        pulses=x_pulses,
    )

    d_status = inspect_batch_matrix(
        name="D",
        signs=d_signs,
        pulses=d_pulses,
    )

    summary["X"][x_status] += 1
    summary["D"][d_status] += 1

    print()


# ============================================================
# Summary
# ============================================================

print("Summary")

for name in ("X", "D"):
    print(
        f"{name}: "
        f"{summary[name]['PERMUTED OK']} permuted, "
        f"{summary[name]['ALIGNED ONLY']} aligned only, "
        f"{summary[name]['INVALID']} invalid"
    )