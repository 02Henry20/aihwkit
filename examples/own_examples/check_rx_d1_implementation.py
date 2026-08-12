from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


BITS_PER_GROUP = 8
DEFAULT_DUMP_FILE = Path(__file__).resolve().parent / "05_repeated_group_train_dump.txt"


# ============================================================
# CLI / loading
# ============================================================


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate an RX-D1 pulse-train dump. RX-D1 means: "
            "D uses one shared LFSR sequence per batch and is row-wise sorted; "
            "X uses independent RNG streams and remains unsorted."
        )
    )
    parser.add_argument(
        "dump_path",
        nargs="?",
        type=Path,
        default=DEFAULT_DUMP_FILE,
        help="RX-D1 train dump to inspect.",
    )
    parser.add_argument(
        "--reference-dump",
        type=Path,
        default=None,
        help=(
            "Optional SORT-D / independent-RNG reference dump with the same "
            "shape/BL. If supplied, statistical metrics are compared directly."
        ),
    )
    parser.add_argument(
        "--old-dump",
        type=Path,
        default=None,
        help=(
            "Optional OLD-S dump. If supplied together with --reference-dump, "
            "the script reports whether RX-D1 is quantitatively closer to the "
            "training-safe reference than OLD-S."
        ),
    )
    parser.add_argument(
        "--pair-samples",
        type=int,
        default=512,
        help="Maximum X-D row pairs sampled per batch for correlation metrics.",
    )
    parser.add_argument(
        "--row-pair-samples",
        type=int,
        default=256,
        help="Maximum X-X and D-D row pairs sampled per batch.",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=4,
        help="Maximum temporal lag checked for residual X-D coupling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Sampling seed used only by this checker.",
    )
    parser.add_argument(
        "--print-trains",
        action="store_true",
        help="Print every reconstructed X/D train.",
    )
    parser.add_argument(
        "--max-print-batches",
        type=int,
        default=4,
        help="Maximum number of batches printed with --print-trains.",
    )
    return parser.parse_args()


def load_train_dump(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, str], dict[str, np.ndarray]]:
    """
    Load the same structured train dump format used by the existing checker.

    Required sections:
        [x_train]
        [d_train]

    Optional sections supported by this checker:
        [x_prob]
        [d_prob]

    If d_prob is available, the checker can perform a much stronger test of the
    shared-D-RNG property because a common random sequence implies monotone pulse
    counts with respect to the D thresholds within each batch.
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Train dump does not exist: {path}")

    metadata: dict[str, str] = {}
    sections: dict[str, list[float]] = {
        "x_train": [],
        "d_train": [],
        "x_prob": [],
        "d_prob": [],
    }
    current_section: str | None = None

    valid_open = {f"[{name}]": name for name in sections}
    valid_close = {f"[/{name}]": name for name in sections}

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line in valid_open:
            current_section = valid_open[line]
            continue

        if line in valid_close:
            expected = valid_close[line]
            if current_section != expected:
                raise ValueError(
                    f"Mismatched section ending at line {line_number}: {line}"
                )
            current_section = None
            continue

        if current_section is not None:
            try:
                if current_section in ("x_train", "d_train"):
                    sections[current_section].extend(int(v) for v in line.split())
                else:
                    sections[current_section].extend(float(v) for v in line.split())
            except ValueError as error:
                raise ValueError(
                    f"Invalid value in [{current_section}] at line {line_number}."
                ) from error
            continue

        if "=" not in line:
            raise ValueError(f"Invalid metadata line {line_number}: {raw_line}")

        key, value = line.split("=", 1)
        metadata[key.strip()] = value.strip()

    if current_section is not None:
        raise ValueError(f"Unclosed section: [{current_section}]")

    required_metadata = {
        "batch_size",
        "input_count",
        "output_count",
        "bl",
        "words_per_train",
        "out_trans",
        "x_train_length",
        "d_train_length",
    }
    missing = required_metadata.difference(metadata)
    if missing:
        raise KeyError(f"Missing metadata field(s): {', '.join(sorted(missing))}")

    x_train = np.asarray(sections["x_train"], dtype=np.uint32)
    d_train = np.asarray(sections["d_train"], dtype=np.uint32)

    if len(x_train) != int(metadata["x_train_length"]):
        raise ValueError("x_train length does not match metadata.")
    if len(d_train) != int(metadata["d_train_length"]):
        raise ValueError("d_train length does not match metadata.")

    optional: dict[str, np.ndarray] = {}
    for name in ("x_prob", "d_prob"):
        if sections[name]:
            optional[name] = np.asarray(sections[name], dtype=np.float64)

    return x_train, d_train, metadata, optional


# ============================================================
# Packed-train reconstruction
# ============================================================


def words_from_bl(bl: int) -> int:
    return (bl + 32) // 32


def get_train(
    train: np.ndarray,
    feature_idx: int,
    batch_idx: int,
    feature_count: int,
    words_per_train: int,
    batch_size: int,
    out_trans: bool,
) -> np.ndarray:
    words = []
    for word_idx in range(words_per_train):
        if out_trans:
            batch_aligned = batch_idx + batch_size * feature_idx
            index = (
                (batch_aligned // feature_count)
                * words_per_train
                * feature_count
                + batch_aligned % feature_count
                + word_idx * feature_count
            )
        else:
            index = (
                feature_idx
                + feature_count * word_idx
                + batch_idx * words_per_train * feature_count
            )
        words.append(train[index])

    return np.asarray(words, dtype=np.uint32)


def unpack_train(words: np.ndarray, bl: int) -> tuple[int, np.ndarray]:
    sign = int(words[0]) & 1
    pulses = np.empty(bl, dtype=np.uint8)

    for pulse_idx in range(bl):
        if pulse_idx < 31:
            word_idx = 0
            bit_idx = pulse_idx + 1
        else:
            shifted_idx = pulse_idx - 31
            word_idx = 1 + shifted_idx // 32
            bit_idx = shifted_idx % 32

        pulses[pulse_idx] = (int(words[word_idx]) >> bit_idx) & 1

    return sign, pulses


def reconstruct_batch(
    train: np.ndarray,
    batch_idx: int,
    feature_count: int,
    words_per_train: int,
    batch_size: int,
    bl: int,
    out_trans: bool,
) -> tuple[np.ndarray, np.ndarray]:
    signs = np.empty(feature_count, dtype=np.uint8)
    pulses = np.empty((feature_count, bl), dtype=np.uint8)

    for feature_idx in range(feature_count):
        words = get_train(
            train,
            feature_idx,
            batch_idx,
            feature_count,
            words_per_train,
            batch_size,
            out_trans,
        )
        signs[feature_idx], pulses[feature_idx] = unpack_train(words, bl)

    return signs, pulses


def reconstruct_all(
    train: np.ndarray,
    feature_count: int,
    words_per_train: int,
    batch_size: int,
    bl: int,
    out_trans: bool,
) -> tuple[np.ndarray, np.ndarray]:
    signs = np.empty((batch_size, feature_count), dtype=np.uint8)
    pulses = np.empty((batch_size, feature_count, bl), dtype=np.uint8)

    for batch_idx in range(batch_size):
        signs[batch_idx], pulses[batch_idx] = reconstruct_batch(
            train,
            batch_idx,
            feature_count,
            words_per_train,
            batch_size,
            bl,
            out_trans,
        )

    return signs, pulses


# ============================================================
# Structural RX-D1 checks
# ============================================================


def is_prefix_train(row: np.ndarray) -> bool:
    """True for 111...1100...000, including all-zero and all-one rows."""
    if row.size <= 1:
        return True
    return bool(np.all(np.diff(row.astype(np.int8)) <= 0))


def prefix_length(row: np.ndarray) -> int:
    zeros = np.flatnonzero(row == 0)
    return int(zeros[0]) if len(zeros) else len(row)


def d_sort_metrics(d: np.ndarray) -> dict[str, float]:
    rows = d.reshape(-1, d.shape[-1])
    prefix_flags = np.asarray([is_prefix_train(row) for row in rows], dtype=bool)

    # For sorted D, each later column mask must be a subset of the previous mask.
    nested_ok = []
    for batch in d:
        if batch.shape[1] <= 1:
            nested_ok.append(True)
        else:
            nested_ok.append(bool(np.all(batch[:, 1:] <= batch[:, :-1])))

    return {
        "d_prefix_row_fraction": float(prefix_flags.mean()) if len(prefix_flags) else np.nan,
        "d_invalid_prefix_rows": int((~prefix_flags).sum()),
        "d_nested_batch_fraction": float(np.mean(nested_ok)) if nested_ok else np.nan,
    }


def x_unsorted_metrics(x: np.ndarray) -> dict[str, float]:
    rows = x.reshape(-1, x.shape[-1])
    counts = rows.sum(axis=1)
    informative = (counts > 0) & (counts < x.shape[-1])

    if informative.any():
        prefix_flags = np.asarray(
            [is_prefix_train(row) for row in rows[informative]],
            dtype=bool,
        )
        prefix_fraction = float(prefix_flags.mean())
        nonprefix_fraction = 1.0 - prefix_fraction
    else:
        prefix_fraction = np.nan
        nonprefix_fraction = np.nan

    return {
        "x_informative_rows": int(informative.sum()),
        "x_prefix_fraction_informative": prefix_fraction,
        "x_nonprefix_fraction_informative": nonprefix_fraction,
    }


# ============================================================
# Pattern / compression metrics
# ============================================================


def pattern_metrics(pulses: np.ndarray) -> dict[str, float]:
    """
    Average metrics across batches. A 'pattern' is a complete time-slice vector
    across all rows of X or D.
    """
    unique_ratios = []
    adjacent_repeat_fractions = []
    zero_column_fractions = []
    entropy_bits = []

    for batch in pulses:
        bl = batch.shape[1]
        columns = [
            tuple(int(bit) for bit in batch[:, t])
            for t in range(bl)
        ]
        counter = Counter(columns)

        unique_ratios.append(len(counter) / bl)

        if bl > 1:
            adjacent_repeat_fractions.append(
                sum(columns[t] == columns[t - 1] for t in range(1, bl))
                / (bl - 1)
            )
        else:
            adjacent_repeat_fractions.append(0.0)

        zero_column_fractions.append(
            sum(not any(column) for column in columns) / bl
        )

        probabilities = np.asarray(list(counter.values()), dtype=np.float64) / bl
        entropy_bits.append(float(-(probabilities * np.log2(probabilities)).sum()))

    return {
        "pattern_diversity_ratio": float(np.mean(unique_ratios)),
        "adjacent_repeat_fraction": float(np.mean(adjacent_repeat_fractions)),
        "zero_column_fraction": float(np.mean(zero_column_fractions)),
        "pattern_entropy_bits": float(np.mean(entropy_bits)),
    }


# ============================================================
# Correlation / dependence metrics
# ============================================================


def pearson_binary(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    sa = a.std()
    sb = b.std()
    if sa == 0.0 or sb == 0.0:
        return np.nan

    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def binary_mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.uint8)
    b = b.astype(np.uint8)

    n = len(a)
    if n == 0:
        return np.nan

    pxy = np.zeros((2, 2), dtype=np.float64)
    for av in (0, 1):
        for bv in (0, 1):
            pxy[av, bv] = np.mean((a == av) & (b == bv))

    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    mi = 0.0
    for av in (0, 1):
        for bv in (0, 1):
            p = pxy[av, bv]
            if p > 0.0 and px[av] > 0.0 and py[bv] > 0.0:
                mi += p * np.log2(p / (px[av] * py[bv]))

    return float(mi)


def sample_pairs(
    n_a: int,
    n_b: int,
    limit: int,
    rng: np.random.Generator,
    same_set: bool = False,
) -> list[tuple[int, int]]:
    if same_set:
        total = n_a * (n_a - 1) // 2
        if total <= limit:
            return [(i, j) for i in range(n_a) for j in range(i + 1, n_a)]

        pairs: set[tuple[int, int]] = set()
        while len(pairs) < limit:
            i = int(rng.integers(0, n_a))
            j = int(rng.integers(0, n_a))
            if i != j:
                if i > j:
                    i, j = j, i
                pairs.add((i, j))
        return list(pairs)

    total = n_a * n_b
    if total <= limit:
        return [(i, j) for i in range(n_a) for j in range(n_b)]

    flat = rng.choice(total, size=limit, replace=False)
    return [(int(k // n_b), int(k % n_b)) for k in flat]


def mean_abs_pair_correlation(
    a: np.ndarray,
    b: np.ndarray,
    pair_limit: int,
    rng: np.random.Generator,
    same_set: bool = False,
) -> tuple[float, int]:
    pairs = sample_pairs(
        a.shape[0],
        b.shape[0],
        pair_limit,
        rng,
        same_set=same_set,
    )

    values = []
    for i, j in pairs:
        r = pearson_binary(a[i], b[j])
        if np.isfinite(r):
            values.append(abs(r))

    return (
        float(np.mean(values)) if values else np.nan,
        len(values),
    )


def xd_metrics_for_batch(
    x: np.ndarray,
    d: np.ndarray,
    pair_limit: int,
    row_pair_limit: int,
    max_lag: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    xd_pairs = sample_pairs(x.shape[0], d.shape[0], pair_limit, rng)

    corr = []
    mi = []
    max_lag_corr = []
    coincidence = []

    for xi, di in xd_pairs:
        xr = x[xi]
        dr = d[di]

        r = pearson_binary(xr, dr)
        if np.isfinite(r):
            corr.append(abs(r))

        mi.append(binary_mutual_information(xr, dr))
        coincidence.append(float(np.mean(xr & dr)))

        lag_values = []
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                xa = xr[-lag:]
                da = dr[: len(dr) + lag]
            elif lag > 0:
                xa = xr[: len(xr) - lag]
                da = dr[lag:]
            else:
                xa = xr
                da = dr

            if len(xa) >= 3:
                lr = pearson_binary(xa, da)
                if np.isfinite(lr):
                    lag_values.append(abs(lr))

        if lag_values:
            max_lag_corr.append(max(lag_values))

    xx, xx_valid = mean_abs_pair_correlation(
        x, x, row_pair_limit, rng, same_set=True
    )
    dd, dd_valid = mean_abs_pair_correlation(
        d, d, row_pair_limit, rng, same_set=True
    )

    return {
        "xd_abs_corr": float(np.mean(corr)) if corr else np.nan,
        "xd_valid_pairs": len(corr),
        "xd_mi_bits": float(np.mean(mi)) if mi else np.nan,
        "xd_max_abs_lag_corr": (
            float(np.mean(max_lag_corr)) if max_lag_corr else np.nan
        ),
        "xd_coincidence_fraction": (
            float(np.mean(coincidence)) if coincidence else np.nan
        ),
        "xx_abs_corr": xx,
        "xx_valid_pairs": xx_valid,
        "dd_abs_corr": dd,
        "dd_valid_pairs": dd_valid,
    }


def aggregate_dependence_metrics(
    x: np.ndarray,
    d: np.ndarray,
    pair_limit: int,
    row_pair_limit: int,
    max_lag: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    batch_metrics = [
        xd_metrics_for_batch(
            x[b],
            d[b],
            pair_limit,
            row_pair_limit,
            max_lag,
            rng,
        )
        for b in range(x.shape[0])
    ]

    output: dict[str, float] = {}
    for key in (
        "xd_abs_corr",
        "xd_mi_bits",
        "xd_max_abs_lag_corr",
        "xd_coincidence_fraction",
        "xx_abs_corr",
        "dd_abs_corr",
    ):
        values = np.asarray([m[key] for m in batch_metrics], dtype=np.float64)
        finite = values[np.isfinite(values)]
        output[key] = float(finite.mean()) if len(finite) else np.nan

    output["xd_valid_pairs"] = int(sum(m["xd_valid_pairs"] for m in batch_metrics))
    output["xx_valid_pairs"] = int(sum(m["xx_valid_pairs"] for m in batch_metrics))
    output["dd_valid_pairs"] = int(sum(m["dd_valid_pairs"] for m in batch_metrics))
    return output


# ============================================================
# Shared-D statistical checks
# ============================================================


def count_coupling_across_batches(d: np.ndarray, pair_limit: int, seed: int) -> dict[str, float]:
    """
    Correlate D row pulse counts across batches. This is only a statistical
    signature of shared-D randomness because the underlying D probabilities may
    themselves be correlated across rows.
    """
    if d.shape[0] < 3:
        return {
            "d_count_corr_across_batches": np.nan,
            "d_count_corr_valid_pairs": 0,
        }

    counts = d.sum(axis=2).T.astype(np.float64)  # [D feature, batch]
    rng = np.random.default_rng(seed + 777)
    pairs = sample_pairs(counts.shape[0], counts.shape[0], pair_limit, rng, same_set=True)

    values = []
    for i, j in pairs:
        a = counts[i]
        b = counts[j]
        if a.std() == 0.0 or b.std() == 0.0:
            continue
        values.append(abs(float(np.corrcoef(a, b)[0, 1])))

    return {
        "d_count_corr_across_batches": float(np.mean(values)) if values else np.nan,
        "d_count_corr_valid_pairs": len(values),
    }


def shared_d_probability_order_test(
    d: np.ndarray,
    d_prob: np.ndarray | None,
) -> dict[str, float]:
    """
    Strong optional RX-D1 test.

    For one common random sequence per batch, if effective pulse thresholds
    satisfy p_i <= p_j, then the generated pulse counts must satisfy k_i <= k_j.
    Sorting changes positions but not row counts.

    This test assumes d_prob stores the effective nonnegative thresholds used by
    pulse generation, or values that are transformed monotonically and equally
    within each batch.
    """
    if d_prob is None:
        return {
            "d_shared_order_comparisons": 0,
            "d_shared_order_consistency": np.nan,
        }

    batch_size, d_size, _ = d.shape
    if d_prob.size != batch_size * d_size:
        raise ValueError(
            "[d_prob] is present but its length is not batch_size * output_count."
        )

    probs = np.abs(d_prob.reshape(batch_size, d_size))
    counts = d.sum(axis=2)

    consistent = 0
    comparisons = 0

    for b in range(batch_size):
        p = probs[b]
        k = counts[b]

        for i in range(d_size):
            for j in range(i + 1, d_size):
                if p[i] == p[j]:
                    continue

                comparisons += 1
                if p[i] < p[j]:
                    consistent += int(k[i] <= k[j])
                else:
                    consistent += int(k[j] <= k[i])

    return {
        "d_shared_order_comparisons": comparisons,
        "d_shared_order_consistency": (
            consistent / comparisons if comparisons else np.nan
        ),
    }


# ============================================================
# Dataset analysis / comparison
# ============================================================


@dataclass
class Analysis:
    name: str
    metadata: dict[str, str]
    structural: dict[str, float]
    x_pattern: dict[str, float]
    d_pattern: dict[str, float]
    dependence: dict[str, float]
    shared_d: dict[str, float]


def analyze_dump(
    name: str,
    path: Path,
    pair_samples: int,
    row_pair_samples: int,
    max_lag: int,
    seed: int,
) -> tuple[Analysis, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, d_train, metadata, optional = load_train_dump(path)

    batch_size = int(metadata["batch_size"])
    x_size = int(metadata["input_count"])
    d_size = int(metadata["output_count"])
    bl = int(metadata["bl"])
    words_per_train = words_from_bl(bl)
    stored_words_per_train = int(metadata["words_per_train"])
    out_trans = bool(int(metadata["out_trans"]))

    if words_per_train != stored_words_per_train:
        raise ValueError(
            f"{name}: BL={bl} requires {words_per_train} words/train, "
            f"dump says {stored_words_per_train}."
        )

    expected_x = x_size * batch_size * words_per_train
    expected_d = d_size * batch_size * words_per_train
    if len(x_train) != expected_x:
        raise ValueError(f"{name}: X dump length {len(x_train)} != expected {expected_x}")
    if len(d_train) != expected_d:
        raise ValueError(f"{name}: D dump length {len(d_train)} != expected {expected_d}")

    x_signs, x = reconstruct_all(
        x_train, x_size, words_per_train, batch_size, bl, out_trans
    )
    d_signs, d = reconstruct_all(
        d_train, d_size, words_per_train, batch_size, bl, out_trans
    )

    structural = {}
    structural.update(d_sort_metrics(d))
    structural.update(x_unsorted_metrics(x))

    x_pattern = pattern_metrics(x)
    d_pattern = pattern_metrics(d)
    dependence = aggregate_dependence_metrics(
        x,
        d,
        pair_samples,
        row_pair_samples,
        max_lag,
        seed,
    )

    shared_d = {}
    shared_d.update(
        count_coupling_across_batches(d, row_pair_samples, seed)
    )
    shared_d.update(
        shared_d_probability_order_test(d, optional.get("d_prob"))
    )

    return (
        Analysis(
            name=name,
            metadata=metadata,
            structural=structural,
            x_pattern=x_pattern,
            d_pattern=d_pattern,
            dependence=dependence,
            shared_d=shared_d,
        ),
        x_signs,
        x,
        d_signs,
        d,
    )


# ============================================================
# Display helpers
# ============================================================


def fmt(value: float, digits: int = 6) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def format_bits(row: np.ndarray) -> str:
    bit_string = "".join(str(int(v)) for v in row)
    return " ".join(
        bit_string[i:i + BITS_PER_GROUP]
        for i in range(0, len(bit_string), BITS_PER_GROUP)
    )


def print_configuration(analysis: Analysis) -> None:
    m = analysis.metadata
    print(f"\n[{analysis.name}]")
    print(f"  batch_size       : {m['batch_size']}")
    print(f"  input_count      : {m['input_count']}")
    print(f"  output_count     : {m['output_count']}")
    print(f"  BL               : {m['bl']}")
    print(f"  out_trans        : {m['out_trans']}")
    if "granularity" in m:
        print(f"  granularity      : {m['granularity']}")


def print_analysis(analysis: Analysis) -> None:
    s = analysis.structural
    xp = analysis.x_pattern
    dp = analysis.d_pattern
    dep = analysis.dependence
    sh = analysis.shared_d

    print("\n=== RX-D1 structural checks ===")
    print(
        f"D rows exact sorted prefix        : {100*s['d_prefix_row_fraction']:.3f}% "
        f"({s['d_invalid_prefix_rows']} invalid)"
    )
    print(
        f"D batches with nested masks       : {100*s['d_nested_batch_fraction']:.3f}%"
    )
    print(f"X informative rows                : {s['x_informative_rows']}")
    print(
        f"X informative rows NOT prefix     : "
        f"{fmt(100*s['x_nonprefix_fraction_informative'], 3)}%"
    )

    print("\n=== Pattern structure ===")
    print("                           X            D")
    print(
        f"Pattern diversity       {xp['pattern_diversity_ratio']:>10.6f}   "
        f"{dp['pattern_diversity_ratio']:>10.6f}"
    )
    print(
        f"Adjacent repeats        {xp['adjacent_repeat_fraction']:>10.6f}   "
        f"{dp['adjacent_repeat_fraction']:>10.6f}"
    )
    print(
        f"Zero columns            {xp['zero_column_fraction']:>10.6f}   "
        f"{dp['zero_column_fraction']:>10.6f}"
    )
    print(
        f"Pattern entropy [bit]   {xp['pattern_entropy_bits']:>10.6f}   "
        f"{dp['pattern_entropy_bits']:>10.6f}"
    )

    print("\n=== Correlation / independence ===")
    print(f"Mean |corr(X,D)|                 : {fmt(dep['xd_abs_corr'])}")
    print(f"Mean X-D mutual information     : {fmt(dep['xd_mi_bits'])} bit")
    print(f"Mean max |lag corr(X,D)|         : {fmt(dep['xd_max_abs_lag_corr'])}")
    print(f"Mean coincidence fraction       : {fmt(dep['xd_coincidence_fraction'])}")
    print(f"Mean |corr(X,X)|                 : {fmt(dep['xx_abs_corr'])}")
    print(f"Mean |corr(D,D)|                 : {fmt(dep['dd_abs_corr'])}")

    print("\n=== Shared-D indicators ===")
    print(
        f"D count corr. across batches      : "
        f"{fmt(sh['d_count_corr_across_batches'])} "
        f"({sh['d_count_corr_valid_pairs']} valid pairs)"
    )

    if sh["d_shared_order_comparisons"] > 0:
        print(
            f"D probability/count order match  : "
            f"{100*sh['d_shared_order_consistency']:.4f}% "
            f"({sh['d_shared_order_comparisons']} comparisons)"
        )
    else:
        print(
            "D probability/count order match  : SKIPPED "
            "(add optional [d_prob] section for the strongest shared-LFSR check)"
        )

    print("\n=== Structural verdict ===")
    d_ok = (
        s["d_prefix_row_fraction"] == 1.0
        and s["d_nested_batch_fraction"] == 1.0
    )
    print(f"D sorted/prefix implementation   : {'PASS' if d_ok else 'FAIL'}")

    x_fraction = s["x_nonprefix_fraction_informative"]
    if not np.isfinite(x_fraction):
        x_status = "INCONCLUSIVE (no informative X rows)"
    elif x_fraction >= 0.50:
        x_status = "PASS"
    else:
        x_status = "WARN (X looks unusually prefix-structured)"
    print(f"X remains unsorted               : {x_status}")

    if sh["d_shared_order_comparisons"] > 0:
        order_ok = sh["d_shared_order_consistency"] >= 0.999
        print(f"Shared-D monotone threshold test : {'PASS' if order_ok else 'FAIL/WARN'}")
    else:
        print("Shared-D monotone threshold test : SKIPPED")


def compare_analyses(current: Analysis, other: Analysis, label: str) -> None:
    print(f"\n=== RX-D1 vs {label} ===")

    metrics = [
        ("|corr(X,D)|", "xd_abs_corr"),
        ("X-D MI [bit]", "xd_mi_bits"),
        ("max |lag corr|", "xd_max_abs_lag_corr"),
        ("|corr(X,X)|", "xx_abs_corr"),
        ("|corr(D,D)|", "dd_abs_corr"),
    ]

    print(f"{'Metric':<22} {'RX-D1':>12} {label:>12} {'Delta':>12}")
    print("-" * 60)
    for display, key in metrics:
        a = current.dependence[key]
        b = other.dependence[key]
        delta = a - b if np.isfinite(a) and np.isfinite(b) else np.nan
        print(
            f"{display:<22} {fmt(a):>12} {fmt(b):>12} {fmt(delta):>12}"
        )

    print("\nPattern diversity")
    print(
        f"  X: RX-D1={current.x_pattern['pattern_diversity_ratio']:.6f}, "
        f"{label}={other.x_pattern['pattern_diversity_ratio']:.6f}"
    )
    print(
        f"  D: RX-D1={current.d_pattern['pattern_diversity_ratio']:.6f}, "
        f"{label}={other.d_pattern['pattern_diversity_ratio']:.6f}"
    )


def print_closeness(current: Analysis, reference: Analysis, old: Analysis) -> None:
    print("\n=== Training-preservation proxy: distance from reference ===")
    keys = [
        "xd_abs_corr",
        "xd_mi_bits",
        "xd_max_abs_lag_corr",
        "xx_abs_corr",
        "dd_abs_corr",
    ]

    def distance(a: Analysis, b: Analysis, include_dd: bool) -> float:
        use_keys = keys if include_dd else keys[:-1]
        total = 0.0
        used = 0
        for key in use_keys:
            av = a.dependence[key]
            bv = b.dependence[key]
            if np.isfinite(av) and np.isfinite(bv):
                total += abs(av - bv)
                used += 1
        return total if used else np.nan

    rx_xd = distance(current, reference, include_dd=False)
    old_xd = distance(old, reference, include_dd=False)
    rx_all = distance(current, reference, include_dd=True)
    old_all = distance(old, reference, include_dd=True)

    print(
        "Cross-operand / X-side distance\n"
        f"  RX-D1 : {fmt(rx_xd)}\n"
        f"  OLD-S : {fmt(old_xd)}"
    )
    if np.isfinite(rx_xd) and np.isfinite(old_xd) and rx_xd > 0:
        print(f"  OLD-S / RX-D1 distance ratio: {old_xd / rx_xd:.3f}x")

    print(
        "\nIncluding intentional D-D structure\n"
        f"  RX-D1 : {fmt(rx_all)}\n"
        f"  OLD-S : {fmt(old_all)}"
    )


def print_trains(
    name: str,
    x_signs: np.ndarray,
    x: np.ndarray,
    d_signs: np.ndarray,
    d: np.ndarray,
    max_batches: int,
) -> None:
    for b in range(min(max_batches, x.shape[0])):
        print(f"\n--- {name} batch {b} ---")
        print("X")
        for i in range(x.shape[1]):
            print(
                f"  X{i:<4} s={int(x_signs[b, i])} {format_bits(x[b, i])}"
            )
        print("D")
        for i in range(d.shape[1]):
            print(
                f"  D{i:<4} s={int(d_signs[b, i])} {format_bits(d[b, i])}"
            )


# ============================================================
# Main
# ============================================================


def main() -> None:
    args = parse_arguments()

    current, xs, x, ds, d = analyze_dump(
        "RX-D1",
        args.dump_path,
        args.pair_samples,
        args.row_pair_samples,
        args.max_lag,
        args.seed,
    )

    print_configuration(current)
    print_analysis(current)

    if args.print_trains:
        print_trains(
            "RX-D1",
            xs,
            x,
            ds,
            d,
            args.max_print_batches,
        )

    reference = None
    if args.reference_dump is not None:
        reference, *_ = analyze_dump(
            "REFERENCE",
            args.reference_dump,
            args.pair_samples,
            args.row_pair_samples,
            args.max_lag,
            args.seed,
        )
        compare_analyses(current, reference, "SORT-D")

    old = None
    if args.old_dump is not None:
        old, *_ = analyze_dump(
            "OLD-S",
            args.old_dump,
            args.pair_samples,
            args.row_pair_samples,
            args.max_lag,
            args.seed,
        )
        compare_analyses(current, old, "OLD-S")

    if reference is not None and old is not None:
        print_closeness(current, reference, old)

    print("\n=== Important limitation ===")
    print(
        "From X/D pulse trains alone, D sorting can be proven exactly and "
        "X/D dependence can be measured statistically. The fact that D used "
        "one shared pre-sort LFSR sequence cannot be proven uniquely after "
        "sorting unless the effective D probabilities/thresholds are also "
        "stored. Add an optional [d_prob] section to enable the monotone "
        "probability-vs-count check."
    )


if __name__ == "__main__":
    main()
