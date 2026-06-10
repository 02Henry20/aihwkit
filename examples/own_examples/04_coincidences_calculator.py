import numpy as np


def read_words(name):
    print(f"\nPaste {name}. Empty line to finish:")
    lines = []

    while True:
        line = input("> ").strip()
        if not line:
            break
        lines.append(line)

    return np.array([int(x) for x in " ".join(lines).split()], dtype=np.uint32)


def words_from_bl(bl):
    """
    C++ uint32 count format:
        bit 0 of first word = sign bit
        remaining bits      = pulse bits

    C++ uses:
        Kplus1 = BL + 1
        nK32 = (Kplus1 + 31) >> 5

    Equivalent:
        nK32 = (BL + 32) // 32
    """
    return (bl + 32) // 32


def get_coincidences(x_words, d_words):
    """
    Reconstructs getNfromCount<one_sided=0, uint32_t>(...)

    C++ logic:
        negative = ((x & 1) ^ (d & 1))

        x_and_d = x & d
        n = popcount(x_and_d)
        n -= x_and_d & 1

        for remaining words:
            n += popcount(x & d)
    """

    x_words = np.array(x_words, dtype=np.uint32)
    d_words = np.array(d_words, dtype=np.uint32)

    if len(x_words) != len(d_words):
        raise ValueError(
            f"x_words and d_words length mismatch: {len(x_words)} vs {len(d_words)}"
        )

    if len(x_words) == 0:
        raise ValueError("Empty x_words/d_words")

    negative = bool((int(x_words[0] ^ d_words[0])) & 1)

    # First word: remove sign bit 0 from coincidence count
    n = int(x_words[0] & d_words[0] & np.uint32(0xFFFFFFFE)).bit_count()

    # Remaining words: all bits are pulse bits
    for k in range(1, len(x_words)):
        n += int(x_words[k] & d_words[k]).bit_count()

    return n, negative


def get_train(train, feature_idx, batch_idx, N, W, B, out_trans):
    """
    Generic reconstruction for X or D count trains.

    This mirrors the C++ getAccCountsDebug lambda:

        if current_out_trans_:
            batch_aligned = i_batch + current_m_batch_ * i
            in_idx = batch_aligned / size * nK32 * size
                   + batch_aligned % size
                   + i_nk * size
        else:
            in_idx = i
                   + size * i_nk
                   + i_batch * nK32 * size

    Python variables:
        feature_idx = i
        batch_idx   = i_batch
        N           = size, either I for X or O for D
        W           = nK32
        B           = m_batch
        word_idx    = i_nk
    """

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


# ============================================================
# Read input
# ============================================================

d_train = read_words("d_train")
x_train = read_words("x_train")

B = int(input("\nBatch size / m_batch: "))
I = int(input("Number of inputs / x_size: "))
O = int(input("Number of outputs / d_size: "))
BL = int(input("BL / current_BL: "))
out_trans = bool(int(input("out_trans, 0 or 1: ")))

W = words_from_bl(BL)

expected_d = O * B * W
expected_x = I * B * W

if len(d_train) != expected_d:
    raise ValueError(f"d_train has {len(d_train)} values, expected {expected_d}")

if len(x_train) != expected_x:
    raise ValueError(f"x_train has {len(x_train)} values, expected {expected_x}")


# ============================================================
# Debug metadata
# ============================================================

print("\n================ META ================")
print(f"B / m_batch       = {B}")
print(f"I / x_size        = {I}")
print(f"O / d_size        = {O}")
print(f"BL / current_BL   = {BL}")
print(f"words_per_train   = {W}")
print(f"out_trans         = {int(out_trans)}")

if out_trans:
    print("\nIndex formula:")
    print("batch_aligned = batch_idx + B * feature_idx")
    print("idx = (batch_aligned // N) * W * N + (batch_aligned % N) + word_idx * N")
else:
    print("\nIndex formula:")
    print("idx = feature_idx + N * word_idx + batch_idx * W * N")

print("\nFor X: N = I")
print("For D: N = O")
print("======================================\n")


# ============================================================
# Reconstruct coincidences
# ============================================================

rows = []

for i in range(I):
    for o in range(O):
        total_abs = 0

        # Device update sign convention:
        # negative == 1 -> + step
        # negative == 0 -> - step
        device_signed_total = 0

        # Algebraic sign convention:
        # negative == 1 -> -n
        # negative == 0 -> +n
        algebraic_signed_total = 0

        negs = []
        batch_ns = []
        batch_device_signed = []
        batch_algebraic_signed = []

        for b in range(B):
            x = get_train(
                train=x_train,
                feature_idx=i,
                batch_idx=b,
                N=I,
                W=W,
                B=B,
                out_trans=out_trans,
            )

            d = get_train(
                train=d_train,
                feature_idx=o,
                batch_idx=b,
                N=O,
                W=W,
                B=B,
                out_trans=out_trans,
            )

            n, neg = get_coincidences(x, d)

            total_abs += n
            batch_ns.append(n)
            negs.append(neg)

            dev_signed = n if neg else -n
            alg_signed = -n if neg else n

            device_signed_total += dev_signed
            algebraic_signed_total += alg_signed

            batch_device_signed.append(dev_signed)
            batch_algebraic_signed.append(alg_signed)

        rows.append(
            [
                f"out{o},in{i}",
                total_abs,
                f"{total_abs / (B * BL):.6f}" if B * BL != 0 else "nan",
                device_signed_total,
                algebraic_signed_total,
                batch_ns,
                negs,
                batch_device_signed,
                batch_algebraic_signed,
            ]
        )


# ============================================================
# Print results
# ============================================================

print("\n================ RESULTS ================\n")

print(
    f"{'Weight':<12} "
    f"{'Total':<8} "
    f"{'Rate':<10} "
    f"{'DevSigned':<10} "
    f"{'AlgSigned':<10} "
    f"{'Batch n':<20} "
    f"{'Negative per batch':<25} "
    f"{'Dev batch':<20} "
    f"{'Alg batch'}"
)

print("-" * 140)

for (
    weight,
    total_abs,
    rate,
    device_signed_total,
    algebraic_signed_total,
    batch_ns,
    negs,
    batch_device_signed,
    batch_algebraic_signed,
) in rows:
    print(
        f"{weight:<12} "
        f"{total_abs:<8} "
        f"{rate:<10} "
        f"{device_signed_total:<10} "
        f"{algebraic_signed_total:<10} "
        f"{str(batch_ns):<20} "
        f"{str(negs):<25} "
        f"{str(batch_device_signed):<20} "
        f"{batch_algebraic_signed}"
    )