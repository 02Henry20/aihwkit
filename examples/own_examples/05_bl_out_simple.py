from __future__ import annotations

import os
from pathlib import Path

os.system("clear")

import numpy as np
import torch

from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import PulseType
from aihwkit.simulator.presets import IdealizedPreset
from aihwkit.simulator.rpu_base import cuda


# =============================================================================
# PARAMETERS
# =============================================================================

BATCH_SIZE = 20
IN_SIZE = 10
OUT_SIZE = 8
BL = 128
GRANULARITY = 0.01

# The script is intended to live in examples/own_examples/.
# The dump is therefore saved in that same directory.
OUTPUT_FILE = Path(__file__).resolve().parent / "05_repeated_group_train_dump.txt"
VALUES_PER_LINE = 16


# =============================================================================
# Model setup
# =============================================================================


def create_stochastic_model(desired_bl: int) -> AnalogLinear:
    rpu_config = IdealizedPreset()

    rpu_config.update.desired_bl = desired_bl
    rpu_config.update.pulse_type = PulseType.STOCHASTIC_COMPRESSED
    rpu_config.update.update_bl_management = False
    rpu_config.update.update_management = False
    rpu_config.update.fixed_bl = True

    rpu_config.device.dw_min = GRANULARITY
    rpu_config.device.dw_min_std = 0
    rpu_config.device.dw_min_dtod = 0
    rpu_config.device.w_max_dtod = 0
    rpu_config.device.w_min_dtod = 0

    rpu_config.forward.is_perfect = True

    return AnalogLinear(
        IN_SIZE,
        OUT_SIZE,
        bias=False,
        rpu_config=rpu_config,
    )


def move_model_to_device(model: AnalogLinear) -> AnalogLinear:
    if cuda.is_compiled():
        return model.cuda()

    print("CUDA not compiled. Running the model on CPU.")
    return model


# =============================================================================
# Train-dump output
# =============================================================================


def write_uint32_section(
    file_handle,
    section_name: str,
    values: np.ndarray,
) -> None:
    """Write one packed uint32 array using short, terminal-safe lines."""
    file_handle.write(f"[{section_name}]\n")

    for start in range(0, len(values), VALUES_PER_LINE):
        chunk = values[start : start + VALUES_PER_LINE]
        file_handle.write(" ".join(str(int(value)) for value in chunk))
        file_handle.write("\n")

    file_handle.write(f"[/{section_name}]\n")


def save_train_dump(
    output_path: Path,
    trains: dict,
    *,
    batch_size: int,
    input_count: int,
    output_count: int,
    bl: int,
    granularity: float,
) -> Path:
    """
    Save metadata, X trains, and D trains in one self-contained text file.
    """
    x_train = np.asarray(trains["x_train"], dtype=np.uint32).reshape(-1)
    d_train = np.asarray(trains["d_train"], dtype=np.uint32).reshape(-1)
    out_trans = int(bool(trains["out_trans"]))

    words_per_train = (bl + 32) // 32
    expected_x_length = batch_size * input_count * words_per_train
    expected_d_length = batch_size * output_count * words_per_train

    if len(x_train) != expected_x_length:
        raise ValueError(
            f"x_train has {len(x_train)} values, "
            f"expected {expected_x_length}."
        )

    if len(d_train) != expected_d_length:
        raise ValueError(
            f"d_train has {len(d_train)} values, "
            f"expected {expected_d_length}."
        )

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file_handle:
        file_handle.write("# AIHWKIT packed pulse-train dump v1\n")
        file_handle.write(f"batch_size={batch_size}\n")
        file_handle.write(f"input_count={input_count}\n")
        file_handle.write(f"output_count={output_count}\n")
        file_handle.write(f"bl={bl}\n")
        file_handle.write(f"words_per_train={words_per_train}\n")
        file_handle.write(f"out_trans={out_trans}\n")
        file_handle.write(f"granularity={granularity:.17g}\n")
        file_handle.write(f"x_train_length={len(x_train)}\n")
        file_handle.write(f"d_train_length={len(d_train)}\n\n")

        write_uint32_section(file_handle, "x_train", x_train)
        file_handle.write("\n")
        write_uint32_section(file_handle, "d_train", d_train)

    temporary_path.replace(output_path)
    return output_path


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if device.type == "cuda":
        print("CUDA version:", torch.version.cuda)
        print("GPU name:", torch.cuda.get_device_name(device))
        print("CUDA compiled:", cuda.is_compiled())
    else:
        print("Running on CPU.")

    model = create_stochastic_model(BL)
    model = move_model_to_device(model)

    x_ref = torch.rand(BATCH_SIZE, IN_SIZE)

    weight_init = model.get_weights(realistic=False)[0]
    weight_target = 2 * torch.rand(OUT_SIZE, IN_SIZE) - 1

    model.set_weights(
        weight=weight_target,
        bias=None,
        realistic=True,
        apply_weight_scaling=False,
        w_init=weight_init.clone(),
        learning_rate=0.2,
        x_values=x_ref,
    )

    weight_after = model.get_weights(realistic=False)[0]
    trains = model.analog_module.tile.get_trains()

    x_train = np.asarray(trains["x_train"], dtype=np.uint32).reshape(-1)
    d_train = np.asarray(trains["d_train"], dtype=np.uint32).reshape(-1)

    # Keep the original console output.
    print("\nx_train:")
    print(" ".join(str(int(value)) for value in x_train))

    print("\nd_train:")
    print(" ".join(str(int(value)) for value in d_train))

    print(f"\nout_trans: {int(bool(trains['out_trans']))}")
    print(f"len x train: {len(x_train)}")
    print(f"len d train: {len(d_train)}")

    print(f"weight_init: {weight_init}\n")
    print(f"weight_tar: {weight_target}")
    print(f"weight_after: {weight_after}\n")
    print(f"granularity: {GRANULARITY}\n")

    saved_path = save_train_dump(
        OUTPUT_FILE,
        trains,
        batch_size=BATCH_SIZE,
        input_count=IN_SIZE,
        output_count=OUT_SIZE,
        bl=BL,
        granularity=GRANULARITY,
    )

    print(f"Saved packed train dump: {saved_path}")


if __name__ == "__main__":
    main()