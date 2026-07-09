import os
import numpy as np
import torch

from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import (
    PulseType,
    UnitCellRPUConfig,
    TransferCompound,
)
from aihwkit.simulator.presets.devices import EcRamPresetDevice
from aihwkit.simulator.rpu_base import cuda


os.system("clear")

BATCH_SIZE = 2
IN_SIZE = 2
OUT_SIZE = 1
BL = 64
GRANULARITY = 0.01


def make_model():
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

    # Main SGD update -> stochastic pulse trains
    cfg.update.desired_bl = BL
    cfg.update.pulse_type = PulseType.STOCHASTIC_COMPRESSED
    cfg.update.fixed_bl = False
    cfg.update.update_bl_management = True
    cfg.update.update_management = True

    # Transfer-read configuration
    cfg.device.transfer_forward = cfg.forward

    # Transfer update configuration
    cfg.device.transfer_update = cfg.update

    model = AnalogLinear(
        IN_SIZE,
        OUT_SIZE,
        bias=False,
        rpu_config=cfg,
    )

    return model.cuda() if cuda.is_compiled() else model


model = make_model()

x = torch.rand(BATCH_SIZE, IN_SIZE)

w_init = model.get_weights(realistic=False)[0]
w_target = 2 * torch.rand(OUT_SIZE, IN_SIZE) - 1

# Single analog weight-setting/update operation.
# No training loop.
model.set_weights(
    weight=w_target,
    bias=None,
    realistic=True,
    apply_weight_scaling=False,
    w_init=w_init.clone(),
    learning_rate=0.2,
    x_values=x,
)

trains = model.analog_module.tile.get_trains()

x_train = np.asarray(
    trains["x_train"],
    dtype=np.uint32,
).reshape(-1)

d_train = np.asarray(
    trains["d_train"],
    dtype=np.uint32,
).reshape(-1)


print(f"CUDA compiled: {cuda.is_compiled()}")
print(f"out_trans: {int(bool(trains['out_trans']))}")
print(f"x_train length: {len(x_train)}")
print(f"d_train length: {len(d_train)}")
print(f"words/train: {(BL + 32) // 32}")
print(f"granularity: {GRANULARITY}")

print("\nx_train:")
print(" ".join(str(int(v)) for v in x_train))

print("\nd_train:")
print(" ".join(str(int(v)) for v in d_train))