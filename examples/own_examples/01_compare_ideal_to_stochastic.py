
import torch
from torch.nn.functional import mse_loss
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, IdealDevice, PulseType,InferenceRPUConfig
from aihwkit.simulator.rpu_base import cuda
from aihwkit.simulator.presets import EcRamPreset,IdealizedPreset
from aihwkit.inference import PCMLikeNoiseModel
import matplotlib.pyplot as plt
import numpy as np


def check_CUDA():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("CUDA version:", torch.version.cuda)
        print("GPU name:", torch.cuda.get_device_name(device))
        print("CUDA Compiled: ",cuda.is_compiled())
    else:
        print("Running on CPU!")

def set_CUDA_model(x,y,model):
    if cuda.is_compiled():
        x = x.cuda()
        y = y.cuda()
        model = model.cuda()


def initialize_same_weight(models):
    weight = torch.Tensor([[-0.5,0.5,-0.5,0.5],[0.5,-0.5,0.5,-0.5]])
    for model in models: 
        model.set_weights(weight=weight.detach().clone(), realistic = True)

def set_same_weights(models):
    torch.manual_seed(100)
    weight = torch.randn(2, 4)
    for model in models: 
        model.set_weights(weight=weight.detach().clone(), realistic = True)
    return weight

def compare_tensors(tensor_ref, tensor_compare, epsilon=1e-8):
    diff = tensor_compare - tensor_ref
    abs_diff = diff.abs()
    
    mean_diff = diff.mean().item()
    std_diff = diff.std().item()
    
    pct_error = abs_diff / (tensor_ref.abs() + epsilon)
    avg_pct_error = pct_error.mean().item() * 100
    
    return mean_diff,std_diff,avg_pct_error


check_CUDA()
x = torch.Tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]])
y = torch.Tensor([[1.0, 0.5], [0.7, 0.3]])

# #Model Ideal
# rpu_config_ideal = IdealizedPreset()
# model_ideal = AnalogLinear(4, 2, bias=False, rpu_config=rpu_config_ideal)
# opt_ideal = AnalogSGD(model_ideal.parameters(), lr=0.1)
# opt_ideal.regroup_param_groups(model_ideal)

#Model Stochastic
rpu_config_stochastic = IdealizedPreset()
rpu_config_stochastic.update.desired_bl = 100
rpu_config_stochastic.update.pulse_type = PulseType.STOCHASTIC_COMPRESSED
rpu_config_stochastic.update.update_bl_management = False
rpu_config_stochastic.update.update_management = False
model_stochastic = AnalogLinear(4, 2, bias=False, rpu_config=rpu_config_stochastic)
opt_stochastic = AnalogSGD(model_stochastic.parameters(), lr=0.1)
opt_stochastic.regroup_param_groups(model_stochastic)
set_CUDA_model(x,y,model_stochastic)


perc_error = np.array([])


for i in range(0,20):
    
    initialize_same_weight([model_stochastic])

    set_weight = set_same_weights([model_stochastic])

    mean_diff,std_diff,avg_pct_error = compare_tensors(\
                            set_weight.flatten(),\
                            model_stochastic.get_weights()[0].flatten())
    
    perc_error = np.append(perc_error,avg_pct_error)

print(f"Mean: {np.mean(perc_error)} %")    
print(f"Std: {np.std(perc_error)} %")

# for epoch in range(1):
#     #ideal
#     opt_ideal.zero_grad()
#     loss_ideal = mse_loss(model_ideal(x), y)
#     loss_ideal.backward()
#     opt_ideal.step()
        
#     #stochastic
#     opt_stochastic.zero_grad()
#     loss_stochastic = mse_loss(model_stochastic(x), y)
#     loss_stochastic.backward()
#     opt_stochastic.step()

# print(model_ideal.get_weights())
# print(model_stochastic.get_weights())

# print(compare_tensors(model_ideal.get_weights()[0].flatten(),\
#     model_stochastic.get_weights()[0].flatten()))