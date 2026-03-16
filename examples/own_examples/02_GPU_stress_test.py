import torch
from torch.nn.functional import mse_loss

from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, ConstantStepDevice
from aihwkit.simulator.rpu_base import cuda

print("AIHWKit CUDA compiled:", cuda.is_compiled())
print("Torch CUDA available:", torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# --- BIG tensors to load GPU ---
batch = 1024
in_features = 4096
out_features = 4096

x = torch.randn(batch, in_features).to(device)
y = torch.randn(batch, out_features).to(device)

# Analog layer
rpu_config = SingleRPUConfig(device=ConstantStepDevice())
model = AnalogLinear(in_features, out_features, bias=True, rpu_config=rpu_config).to(device)

opt = AnalogSGD(model.parameters(), lr=0.01)
opt.regroup_param_groups(model)

print("Model device:", next(model.parameters()).device)

# --- Training ---
epochs = 200

torch.cuda.synchronize()
import time
start = time.time()

for i in range(epochs):
    opt.zero_grad()

    pred = model(x)
    loss = mse_loss(pred, y)

    loss.backward()
    opt.step()

    if i % 20 == 0:
        print(f"Epoch {i}, Loss {loss.item():.4f}")

torch.cuda.synchronize()
end = time.time()

print("Training time:", end-start, "seconds")
print("GPU memory used:", torch.cuda.memory_allocated()/1e9, "GB")