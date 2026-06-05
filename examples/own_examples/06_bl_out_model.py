import os
import numpy as np
from aihwkit.simulator.presets.configs import EcRamPreset

os.system("clear")

import torch
from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.simulator.configs import PulseType
from aihwkit.simulator.rpu_base import cuda
from aihwkit.simulator.presets import IdealizedPreset

from torch import Tensor
from torch.nn.functional import mse_loss

from aihwkit.optim import AnalogSGD

BATCH_SIZE = 200
IN_SIZE = 64
OUT_SIZE = 5
BL = 500
GRANULARITY = 0.0005

def tensors_same(a, b):
    a = a.detach().cpu()
    b = b.detach().cpu()

    same_shape = a.shape == b.shape
    same_content = same_shape and torch.equal(a, b)


    return same_content and same_shape

def trains_to_signed_n_tensor(trains, K_out=None):


    x_train = np.array(trains["x_train"], dtype=np.uint32)
    d_train = np.array(trains["d_train"], dtype=np.uint32)

    out_trans = bool(trains["out_trans"])
    I = int(trains["x_size"])
    O = int(trains["d_size"])
    B = len(K_out)

    if len(x_train) % (I * B) != 0:
        raise ValueError(
            f"Cannot infer W: len(x_train)={len(x_train)}, I={I}, B={B}"
        )

    W = len(x_train) // (I * B)

    expected_x = I * B * W
    expected_d = O * B * W

    if len(x_train) != expected_x:
        raise ValueError(f"x_train has {len(x_train)} values, expected {expected_x}")

    if len(d_train) != expected_d:
        raise ValueError(f"d_train has {len(d_train)} values, expected {expected_d}")

    def get_train(train, feature_idx, batch_idx, N):
        words = []

        for word_idx in range(W):
            if out_trans:
                batch_aligned = batch_idx + B * feature_idx
                idx = (
                    (batch_aligned // N) * W * N
                    + (batch_aligned % N)
                    + word_idx * N
                )
            else:
                idx = (
                    feature_idx
                    + N * word_idx
                    + batch_idx * W * N
                )

            words.append(train[idx])

        return np.array(words, dtype=np.uint32)

    def get_coincidences(x_words, d_words):
        negative = bool((int(x_words[0] ^ d_words[0])) & 1)

        # First word: exclude sign bit 0.
        n = int(x_words[0] & d_words[0] & np.uint32(0xFFFFFFFE)).bit_count()

        # Remaining words: all bits are pulse bits.
        for k in range(1, len(x_words)):
            n += int(x_words[k] & d_words[k]).bit_count()

        return n, negative

    result = torch.zeros((O, I), dtype=torch.int32)

    for o in range(O):
        for i in range(I):
            device_signed_total = 0

            for b in range(B):
                x = get_train(x_train, feature_idx=i, batch_idx=b, N=I)
                d = get_train(d_train, feature_idx=o, batch_idx=b, N=O)

                n, negative = get_coincidences(x, d)

                # Device convention:
                # negative == 1 -> +n
                # negative == 0 -> -n
                device_signed_total += n if negative else -n

            result[o, i] = device_signed_total

    return result


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
        [0.0,0.1],
        [0.2,0.3]
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
    rpu_config.device.dw_min = GRANULARITY
    rpu_config.update.fixed_bl = True
    rpu_config.device.dw_min_std = 0
    rpu_config.device.dw_min_dtod = 0
    rpu_config.device.w_max_dtod = 0
    rpu_config.device.w_min_dtod = 0
    rpu_config.forward.is_perfect = True
    model = AnalogSequential(
        AnalogLinear(IN_SIZE, 64, bias=False, rpu_config=rpu_config),
        AnalogLinear(64, 16, bias=False, rpu_config=rpu_config),
        AnalogLinear(16, OUT_SIZE, bias=False, rpu_config=rpu_config),
    )

    return model

def move_model_to_device(model):
    if cuda.is_compiled():
      return model.cuda()
    else:
      print("Cuda not Compiled")
      return model
    
    


# Prepare the datasets (input and expected output).
x = torch.rand(BATCH_SIZE, IN_SIZE)
y = torch.rand(BATCH_SIZE, OUT_SIZE)


model = create_stochastic_model(100)
# Move the model and tensors to cuda if it is available.
if cuda.is_compiled():
    x = x.cuda()
    y = y.cuda()
    model = model.cuda()

# Define an analog-aware optimizer, preparing it for using the layers.
opt = AnalogSGD(model.parameters(), lr=0.1)
opt.regroup_param_groups(model)

# print(f"{model.get_weights()}")

def _clone_weight_pair(weight_pair):
    """
    Clone a single AIHWKit (weight, bias) tuple onto CPU.

    Storing snapshots on CPU avoids device mismatches when the analog tile
    returns CPU tensors while the ideal model is on CUDA.
    """
    weight, bias = weight_pair

    return (
        weight.detach().cpu().clone(),
        None if bias is None else bias.detach().cpu().clone()
        )


from collections.abc import Iterable
from tqdm import tqdm
import torch
import numpy as np


def iter_analog_tiles(model):
    """
    Yields:
        name, analog_module, tile
    """
    modules = model if isinstance(model, Iterable) else [model]

    for module_idx, module in enumerate(modules):
        if not hasattr(module, "analog_module"):
            continue

        analog = module.analog_module

        if hasattr(analog, "tile"):
            yield f"module_{module_idx}", analog, analog.tile

        elif hasattr(analog, "array"):
            for r, row in enumerate(analog.array):
                for c, tile_wrapper in enumerate(row):
                    yield f"module_{module_idx}_tile_{r}_{c}", analog, tile_wrapper.tile


for epoch in tqdm(range(1), desc="Epochs", leave=True):

    weights_before = {}

    # Save weights before update
    for name, analog, tile in iter_analog_tiles(model):
        if hasattr(analog, "get_weights"):
            weights_before[name] = _clone_weight_pair(analog.get_weights())

    opt.zero_grad()

    pred = model(x)
    loss = mse_loss(pred, y)

    loss.backward()
    opt.step()

    issue_found = False

    # Compare tile by tile after update
    for name, analog, tile in iter_analog_tiles(model):
        if not hasattr(analog, "get_weights"):
            continue

        w_before, b_before = weights_before[name]
        w_after, b_after = _clone_weight_pair(analog.get_weights())

        n_actual = torch.round(
            (w_after - w_before) / GRANULARITY
        ).to(torch.int32)

        n_calc = trains_to_signed_n_tensor(
            tile.get_trains(),
            tile.get_K_out()
        )
        print(f"actual:{n_actual}")
        print(f"calc:{n_calc}")
        same = tensors_same(n_actual, n_calc)

        if not same:
            print("ISSUE!!")
            print(f"epoch: {epoch}")
            print(f"tile/module: {name}")

            print(f"n_actual shape: {n_actual.shape}")
            print(f"n_calc shape:   {n_calc.shape}")

            print(f"n_actual:\n{n_actual}")
            print(f"n_calc:\n{n_calc}")

            issue_found = True
            break

    if issue_found:
        break