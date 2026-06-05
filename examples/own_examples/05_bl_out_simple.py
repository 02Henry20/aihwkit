import os

from aihwkit.simulator.presets.configs import EcRamPreset

os.system("clear")

import torch
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import PulseType
from aihwkit.simulator.rpu_base import cuda
from aihwkit.simulator.presets import IdealizedPreset
import numpy as np
import torch
BATCH_SIZE = 7
IN_SIZE = 4
OUT_SIZE = 3
BL = 128
GRANULARITY = 0.001
def tensors_same(a, b, verbose=True):
    a = a.detach().cpu()
    b = b.detach().cpu()

    same_shape = a.shape == b.shape
    same_content = same_shape and torch.equal(a, b)

    if verbose:
        print(f"same shape: {same_shape}")
        print(f"same content: {same_content}")

        if not same_content:
            print("\na:")
            print(a)
            print("\nb:")
            print(b)

            if same_shape:
                print("\ndiff = a - b:")
                print(a - b)

    return same_content

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
        [0.0,0.1]
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
    model = AnalogLinear(IN_SIZE, OUT_SIZE, bias=False, rpu_config=rpu_config)

    return model

def move_model_to_device(model):
    if cuda.is_compiled():
      return model.cuda()
    else:
      print("Cuda not Compiled")
      return model
    
    
   
bl = BL
model = create_stochastic_model(bl)
model = move_model_to_device(model)
# initialize_same_weight(model)


x_ref1 = torch.rand(BATCH_SIZE, IN_SIZE)


x_ref2 = torch.tensor([
    [0.5, 0.4],
    # [0.1, 0.3],
    # [0.2, 0.2],
    # [0.3, 0.1],
    # [0.4, 0.42]
])


weight_init = model.get_weights(realistic=False)[0]

weight_tar = x = 2 *torch.rand(OUT_SIZE, IN_SIZE)- 1
print("\n\n")
model.set_weights(weight=weight_tar,
                  bias=None,
                  realistic = True,
                  apply_weight_scaling =False,
                  w_init=weight_init.clone(),
                  learning_rate = 0.2,
                  x_values = x_ref1)

# print("\n\n")
# model.set_weights(weight=weight_tar,
#                   bias=None,
#                   realistic = True,
#                   apply_weight_scaling =False,
#                   w_init=weight_init.clone(),
#                   learning_rate = 0.2,
#                   x_values = x_ref2)


weight_after = model.get_weights(realistic=False)[0]

weight_error_before = weight_init - weight_tar
weight_error_after = weight_after - weight_tar

n_actual = (weight_after - weight_init) / GRANULARITY

# print(f"K_out: {model.analog_module.tile.get_K_out()}")
# print(f"Trains: {model.analog_module.tile.get_trains()}")
print(f"weight_init: {weight_init}\n")
print(f"weight_tar: {weight_tar}")
print(f"weight_after: {weight_after}\n")


print(f"granularity: {GRANULARITY}\n")


same = tensors_same(
    trains_to_signed_n_tensor(
        model.analog_module.tile.get_trains(),
        model.analog_module.tile.get_K_out()
    ),
    torch.round(n_actual).to(torch.int32)
)
