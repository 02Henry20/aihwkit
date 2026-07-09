from pathlib import Path

import numpy as np


BITS_PER_GROUP = 8
MAX_BATCHES_TO_PRINT = 5

EXPECT_X_LEFT_ALIGNED = False
EXPECT_D_LEFT_ALIGNED = True

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

    return {
        "path": path,
        "metadata": metadata,
        "x_train": read_uint32_section(lines, "x_train"),
        "d_train": read_uint32_section(lines, "d_train"),
        "B": int(metadata["batch_size"]),
        "I": int(metadata["input_count"]),
        "O": int(metadata["output_count"]),
        "BL": int(metadata["bl"]),
        "out_trans": bool(int(metadata["out_trans"])),
    }


def words_from_bl(bl):
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
    for idx in range(len(pulses) - 1):
        if pulses[idx] == 0 and pulses[idx + 1] == 1:
            return False, idx

    return True, None


def format_bits(pulses):
    bit_string = "".join(str(int(bit)) for bit in pulses)

    return " ".join(
        bit_string[start : start + BITS_PER_GROUP]
        for start in range(0, len(bit_string), BITS_PER_GROUP)
    )


def inspect_train(
    name,
    feature_idx,
    batch_idx,
    packed_words,
    bl,
    expect_left_aligned,
):
    sign, pulses = unpack_train(packed_words, bl)
    aligned, violation_idx = check_left_alignment(pulses)

    if expect_left_aligned:
        status = "OK" if aligned else "FAIL"
    else:
        status = "ALGN" if aligned else "RAND"

    print(
        f"B{batch_idx:<5} "
        f"{name}{feature_idx:<3} "
        f"sign={sign} "
        f"ones={int(pulses.sum()):>3}/{bl:<3} "
        f"{status:<4} "
        f"{format_bits(pulses)}"
    )

    if expect_left_aligned and not aligned:
        print(
            f"        first invalid 0 -> 1 transition: "
            f"pulse {violation_idx} -> {violation_idx + 1}"
        )

    return aligned


def print_metadata(dump):
    meta = dump["metadata"]

    print(f"Loaded dump: {dump['path']}")
    print(f"preset: {meta.get('preset', 'unknown')}")
    print(f"out_trans: {int(dump['out_trans'])}")
    print(f"BL: {dump['BL']}")
    print(f"words/train: {words_from_bl(dump['BL'])}")

    if "original_batch_size" in meta:
        print("\nDetected CNN-style dump")
        print(f"original_batch_size: {meta.get('original_batch_size')}")
        print(f"image_size: {meta.get('image_size')}")
        print(f"in_channels: {meta.get('in_channels')}")
        print(f"out_channels: {meta.get('out_channels')}")
        print(f"kernel_size: {meta.get('kernel_size')}")
        print(f"stride: {meta.get('stride')}")
        print(f"padding: {meta.get('padding')}")
    else:
        print("\nDetected FC-style dump")

    print("\nEffective update dimensions")
    print(f"B = {dump['B']}")
    print(f"I = {dump['I']}")
    print(f"O = {dump['O']}")


def validate_lengths(x_train, d_train, B, I, O, W):
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


def main():
    dump = load_train_dump()

    x_train = dump["x_train"]
    d_train = dump["d_train"]

    B = dump["B"]
    I = dump["I"]
    O = dump["O"]
    BL = dump["BL"]
    out_trans = dump["out_trans"]
    W = words_from_bl(BL)

    validate_lengths(x_train, d_train, B, I, O, W)
    print_metadata(dump)

    x_failures = []
    d_failures = []

    print("\nExpected D order: 111...000")
    print("X is not required to be left-aligned unless EXPECT_X_LEFT_ALIGNED=True.")
    print("Bits are printed from pulse 0 to pulse BL-1.\n")

    batches_to_print = min(B, MAX_BATCHES_TO_PRINT)

    for batch_idx in range(B):
        should_print = batch_idx < batches_to_print

        if should_print:
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

            sign, pulses = unpack_train(words, BL)
            aligned, _ = check_left_alignment(pulses)

            if should_print:
                inspect_train(
                    name="X",
                    feature_idx=input_idx,
                    batch_idx=batch_idx,
                    packed_words=words,
                    bl=BL,
                    expect_left_aligned=EXPECT_X_LEFT_ALIGNED,
                )

            if EXPECT_X_LEFT_ALIGNED and not aligned:
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

            sign, pulses = unpack_train(words, BL)
            aligned, _ = check_left_alignment(pulses)

            if should_print:
                inspect_train(
                    name="D",
                    feature_idx=output_idx,
                    batch_idx=batch_idx,
                    packed_words=words,
                    bl=BL,
                    expect_left_aligned=EXPECT_D_LEFT_ALIGNED,
                )

            if EXPECT_D_LEFT_ALIGNED and not aligned:
                d_failures.append((batch_idx, output_idx))

        if should_print:
            print()

    if B > batches_to_print:
        print(f"... skipped printing {B - batches_to_print} remaining batches\n")

    total_x = B * I
    total_d = B * O

    print("Summary")

    if EXPECT_X_LEFT_ALIGNED:
        print(f"X: {total_x - len(x_failures)}/{total_x} left-aligned")
    else:
        print("X: left-alignment not required")

    if EXPECT_D_LEFT_ALIGNED:
        print(f"D: {total_d - len(d_failures)}/{total_d} left-aligned")
    else:
        print("D: left-alignment not required")

    if d_failures:
        print(
            "Failed D: "
            + ", ".join(
                f"B{batch_idx}/D{output_idx}"
                for batch_idx, output_idx in d_failures[:20]
            )
        )

        if len(d_failures) > 20:
            print(f"... plus {len(d_failures) - 20} more D failures")


if __name__ == "__main__":
    main()