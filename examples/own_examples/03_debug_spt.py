
import torch
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import PulseType
from aihwkit.simulator.rpu_base import cuda
from aihwkit.simulator.presets import IdealizedPreset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("CUDA version:", torch.version.cuda)
    print("GPU name:", torch.cuda.get_device_name(device))
    print("CUDA Compiled: ",cuda.is_compiled())
else:
    print("Running on CPU!")

x = torch.Tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]])
y = torch.Tensor([[1.0, 0.5], [0.7, 0.3]])

rpu_config = IdealizedPreset()
rpu_config.update.desired_bl = 100
rpu_config.update.pulse_type = PulseType.STOCHASTIC_COMPRESSED
rpu_config.update.update_bl_management = False
rpu_config.update.update_management = False
model = AnalogLinear(4, 2, bias=False, rpu_config=rpu_config)

if cuda.is_compiled():
    x = x.cuda()
    y = y.cuda()
    model = model.cuda()


torch.manual_seed(100)
weight = torch.randn(2, 4)
print("\n\n")
model.set_weights(weight=weight.clone(), realistic = True)
