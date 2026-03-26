
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

def initialize_same_weight(model):
    weight = torch.tensor([
        [0.0]
    ])
    
    model.set_weights(weight=weight.clone(), realistic = False)
    if not torch.equal(model.get_weights(realistic = False)[0], weight):
        return False
    else:
        return True    

            
def create_stochastic_model(desired_bl):
    rpu_config = IdealizedPreset()

    rpu_config.update.desired_bl = desired_bl
    rpu_config.update.pulse_type = PulseType.STOCHASTIC_COMPRESSED
    rpu_config.update.update_bl_management = False
    rpu_config.update.update_management = False
    rpu_config.update.fixed_bl = True
    rpu_config.device.dw_min_std = 0
    rpu_config.device.dw_min_dtod = 0
    rpu_config.device.w_max_dtod = 0
    rpu_config.device.w_min_dtod = 0
    rpu_config.forward.is_perfect = True
    model = AnalogLinear(1, 1, bias=False, rpu_config=rpu_config)

    return model

def move_model_to_device(model):
    if cuda.is_compiled():
      return model.cuda()
    else:
      print("Cuda not Compiled")
      return model
    
# 1183 is max    
bl = 100
model = create_stochastic_model(bl)
model = move_model_to_device(model)
initialize_same_weight(model)
x_ref = torch.tensor([1.0])

weight_init = model.get_weights(realistic=False)[0]

weight_tar = torch.tensor([0.1])
print("\n\n")


model.set_weights(weight=weight_tar, bias=None, realistic = True, apply_weight_scaling =False,w_init=weight_init.clone(), learning_rate = 0.1, x_values = x_ref)
weight_after = model.get_weights(realistic=False)[0]

weight_error = (weight_after-weight_tar)/weight_tar

print(f"weight_init: {weight_init}\n")
print(f"weight_tar: {weight_tar}")
print(f"weight_after: {weight_after}\n")
print(f"weight_error {weight_error}")
