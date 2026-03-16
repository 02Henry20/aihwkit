# -*- coding: utf-8 -*-
# (C) Copyright 2020-2024 IBM. All Rights Reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 1: simple network with one layer.
Modified to confirm whether GPU is actually being used.
"""

# Imports from PyTorch.
from torch import Tensor
from torch.nn.functional import mse_loss
import torch

# Imports from aihwkit.
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, ConstantStepDevice
from aihwkit.simulator.rpu_base import cuda

# Prepare the datasets (input and expected output).
x = Tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]])
y = Tensor([[1.0, 0.5], [0.7, 0.3]])

# --- Determine device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("CUDA version:", torch.version.cuda)
    print("GPU name:", torch.cuda.get_device_name(device))
else:
    print("Running on CPU!")

# Define a single-layer network, using a constant step device type.
rpu_config = SingleRPUConfig(device=ConstantStepDevice())
model = AnalogLinear(4, 2, bias=True, rpu_config=rpu_config)

# Move the model and tensors to the selected device
x = x.to(device)
y = y.to(device)
model = model.to(device)

# Define an analog-aware optimizer
opt = AnalogSGD(model.parameters(), lr=0.1)
opt.regroup_param_groups(model)

# --- Check where model parameters reside ---
param_device = next(model.parameters()).device
print("Model parameters device:", param_device)

# Training loop
for epoch in range(100):
    opt.zero_grad()
    pred = model(x)
    loss = mse_loss(pred, y)
    loss.backward()
    opt.step()
    
    # Check that output is on GPU
    print(f"Epoch {epoch+1:03d}, Loss: {loss:.6f}, Output device: {pred.device}")

# Optional: perform a quick timing test to confirm GPU is doing work
if device.type == "cuda":
    torch.cuda.synchronize()
    import time
    start = time.time()
    for _ in range(1000):
        _ = model(x)
    torch.cuda.synchronize()
    end = time.time()
    print("Time for 1000 forward passes on GPU:", end-start, "seconds")