from __future__ import annotations

import os
from pathlib import Path

os.system("clear")

import numpy as np
import torch
import torch.nn.functional as F

from aihwkit.nn import AnalogConv2d
from aihwkit.simulator.configs import PulseType, UnitCellRPUConfig, TransferCompound
from aihwkit.simulator.presets import IdealizedPreset, EcRamPreset
from aihwkit.simulator.presets.devices import EcRamPresetDevice
from aihwkit.simulator.rpu_base import cuda


# =============================================================================
# PARAMETERS
# =============================================================================

BATCH_SIZE = 128
IMG_SIZE = 28
IN_CHANNELS = 1
OUT_CHANNELS = 10
KERNEL_SIZE = 4
STRIDE = 1
PADDING = 0
BL = 64
GRANULARITY = 0.01

# PRESET = "ECRAM"
# PRESET = "IDEALIZED"
PRESET = "TT_ECRAM"

OUTPUT_FILE = Path(__file__).resolve().parent / "05_repeated_group_train_dump.txt"
VALUES_PER_LINE = 16


# =============================================================================
# RPU configs
# =============================================================================


def configure_update(cfg, desired_bl: int) -> None:
    cfg.update.desired_bl = desired_bl
    cfg.update.pulse_type = PulseType.STOCHASTIC_COMPRESSED
    cfg.update.fixed_bl = False
    cfg.update.update_bl_management = True
    cfg.update.update_management = True


def create_config_idealized(desired_bl: int):
    cfg = IdealizedPreset()

    cfg.update.desired_bl = desired_bl
    cfg.update.pulse_type = PulseType.STOCHASTIC_COMPRESSED
    cfg.update.fixed_bl = False
    cfg.update.update_bl_management = True
    cfg.update.update_management = True

    cfg.device.dw_min = GRANULARITY
    cfg.device.dw_min_std = 0
    cfg.device.dw_min_dtod = 0
    cfg.device.w_max_dtod = 0
    cfg.device.w_min_dtod = 0

    cfg.forward.is_perfect = True

    return cfg


def create_config_ecram(desired_bl: int):
    cfg = EcRamPreset()
    configure_update(cfg, desired_bl)
    cfg.device.dw_min = GRANULARITY
    return cfg


def create_config_tt_ecram(desired_bl: int):
    cfg = UnitCellRPUConfig(
        device=TransferCompound(
            unit_cell_devices=[
                EcRamPresetDevice(dw_min=GRANULARITY),
                EcRamPresetDevice(dw_min=GRANULARITY),
            ],
            units_in_mbatch=True,
            transfer_every=2,
            n_reads_per_transfer=1,
            gamma=0.0,
            scale_transfer_lr=True,
            transfer_lr=1.0,
            fast_lr=0.1,
            transfer_columns=True,
        )
    )

    configure_update(cfg, desired_bl)

    cfg.device.transfer_forward = cfg.forward
    cfg.device.transfer_update = cfg.update

    return cfg


def create_rpu_config(desired_bl: int):
    if PRESET == "IDEALIZED":
        return create_config_idealized(desired_bl)

    if PRESET == "ECRAM":
        return create_config_ecram(desired_bl)

    if PRESET == "TT_ECRAM":
        return create_config_tt_ecram(desired_bl)

    raise ValueError(f"Unknown PRESET: {PRESET}")


# =============================================================================
# Model setup
# =============================================================================


def create_stochastic_cnn_layer(desired_bl: int) -> AnalogConv2d:
    rpu_config = create_rpu_config(desired_bl)

    layer = AnalogConv2d(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        kernel_size=KERNEL_SIZE,
        stride=STRIDE,
        padding=PADDING,
        bias=False,
        rpu_config=rpu_config,
    )

    return layer.cuda() if cuda.is_compiled() else layer


def unfold_conv_input(x: torch.Tensor) -> torch.Tensor:
    patches = F.unfold(
        x,
        kernel_size=KERNEL_SIZE,
        stride=STRIDE,
        padding=PADDING,
    )

    patches = patches.transpose(1, 2)

    return patches.reshape(
        -1,
        IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE,
    )


def get_first_tile(layer: AnalogConv2d):
    return next(layer.analog_tiles()).tile


# =============================================================================
# Train-dump output
# =============================================================================


def write_uint32_section(file_handle, section_name: str, values: np.ndarray) -> None:
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
    x_train = np.asarray(trains["x_train"], dtype=np.uint32).reshape(-1)
    d_train = np.asarray(trains["d_train"], dtype=np.uint32).reshape(-1)
    out_trans = int(bool(trains["out_trans"]))

    words_per_train = (bl + 32) // 32

    expected_x_length = batch_size * input_count * words_per_train
    expected_d_length = batch_size * output_count * words_per_train

    if len(x_train) != expected_x_length:
        raise ValueError(
            f"x_train has {len(x_train)} values, expected {expected_x_length}."
        )

    if len(d_train) != expected_d_length:
        raise ValueError(
            f"d_train has {len(d_train)} values, expected {expected_d_length}."
        )

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file_handle:
        file_handle.write("# AIHWKIT packed pulse-train dump v1\n")
        file_handle.write(f"preset={PRESET}\n")
        file_handle.write(f"original_batch_size={BATCH_SIZE}\n")
        file_handle.write(f"image_size={IMG_SIZE}\n")
        file_handle.write(f"in_channels={IN_CHANNELS}\n")
        file_handle.write(f"out_channels={OUT_CHANNELS}\n")
        file_handle.write(f"kernel_size={KERNEL_SIZE}\n")
        file_handle.write(f"stride={STRIDE}\n")
        file_handle.write(f"padding={PADDING}\n")

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

    layer = create_stochastic_cnn_layer(BL)

    x_img = torch.rand(BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE)

    if cuda.is_compiled():
        x_img = x_img.cuda()

    x_patches = unfold_conv_input(x_img)

    weight_init = layer.get_weights(realistic=False)[0]
    weight_target = 2 * torch.rand_like(weight_init) - 1

    layer.set_weights(
        weight=weight_target,
        bias=None,
        realistic=True,
        apply_weight_scaling=False,
        w_init=weight_init.clone(),
        learning_rate=0.2,
        x_values=x_patches,
    )

    trains = get_first_tile(layer).get_trains()

    x_train = np.asarray(trains["x_train"], dtype=np.uint32).reshape(-1)
    d_train = np.asarray(trains["d_train"], dtype=np.uint32).reshape(-1)

    out_size = (IMG_SIZE + 2 * PADDING - KERNEL_SIZE) // STRIDE + 1

    effective_batch_size = BATCH_SIZE * out_size * out_size
    effective_input_count = IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE
    effective_output_count = OUT_CHANNELS

    print(f"\npreset: {PRESET}")
    print(f"out_trans: {int(bool(trains['out_trans']))}")
    print(f"len x train: {len(x_train)}")
    print(f"len d train: {len(d_train)}")
    print(f"granularity: {GRANULARITY}")

    print("\nCNN effective dimensions:")
    print(f"effective_batch_size: {effective_batch_size}")
    print(f"effective_input_count: {effective_input_count}")
    print(f"effective_output_count: {effective_output_count}")
    print(f"x_patches shape: {tuple(x_patches.shape)}")
    print(f"weight_target shape: {tuple(weight_target.shape)}")

    saved_path = save_train_dump(
        OUTPUT_FILE,
        trains,
        batch_size=effective_batch_size,
        input_count=effective_input_count,
        output_count=effective_output_count,
        bl=BL,
        granularity=GRANULARITY,
    )

    print(f"\nSaved packed train dump: {saved_path}")


if __name__ == "__main__":
    main()