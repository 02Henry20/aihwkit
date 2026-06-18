import torch
from torch import Tensor
from torch.nn.functional import mse_loss

from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.presets.configs import EcRamPreset
from aihwkit.simulator.configs import PulseType
from aihwkit.simulator.rpu_base import cuda
import os

import inspect

os.system("clear")


def create_config(bl):
    rpu_config = EcRamPreset()
    rpu_config.update.desired_bl = bl
    rpu_config.update.pulse_type = PulseType.STOCHASTIC_COMPRESSED
    rpu_config.update.update_bl_management = True
    rpu_config.update.update_management = True
    rpu_config.update.fixed_bl = False
    rpu_config.device.dw_min = 0.01
    return rpu_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = Tensor([[0.1, 0.2],[0.4, 0.3],[0.2, 0.1]]) 
y = Tensor([[0.3],[0.1],[0.4]])
model = AnalogLinear(2, 1, bias=False, rpu_config=create_config(100))

if cuda.is_compiled():
    x, y, model = x.cuda(), y.cuda(), model.cuda()

opt = AnalogSGD(model.parameters(), lr=0.1)
opt.regroup_param_groups(model)

# for tile in model.analog_tiles():
#     print(tile)
# print("\n")
# print(model.analog_module)

model.analog_module.tile.set_k_scheduler(int(100))

for epoch in range(1,20):
    opt.zero_grad()
    loss = mse_loss(model(x), y)
    loss.backward()
    opt.step()
    print(f"K_out: {model.analog_module.tile.get_K_out()}")
