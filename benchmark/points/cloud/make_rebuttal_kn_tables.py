#!/usr/bin/env python3

import argparse
import math
import re
from pathlib import Path

import pandas as pd


# 95% two-sided t critical values for small n
# good enough for rebuttal-time seed aggregation
T_CRIT_95 = {
    1: float("nan"),
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
}


FILENAME_RE = re.compile(
    r"^umc_kn_sweep_"
    r"(?P<dataset>[^_]+)_"
    r"(?P<axis>[KN])_"
    r"K(?P<K>\d+)_"
    r"N(?P<N>\d+)_"
    r"train-(?P<train_mode>[^_]+)_"
    r"lambda(?P<lambda_ortho>[^_]+)_"
    r"methods-(?P<methods>[^_]+)_"
    r"seeds-(?P<seeds>.+)\.csv$"
)


def ci95(series: pd.Series) -> float:
    vals = series.dropna().astype(float).tolist()
    n = len(vals)
    if n <= 1:
        return float("nan")
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
    sd = math.sqrt(var)
    tcrit = T_CRIT_95.get(n, 1.96)  # fallback to normal approx for larger n
    return tcrit * sd / math.sqrt(n)


def fmt_pct(mean: float, ci: float, digits: int = 2) -> str:
    if pd.isna(mean):
        return "—"
    mean_pct = 100.0 * mean
    if pd.isna(ci):
        return f"{mean_pct:.{digits}f}"
    ci_pct = 100.0 * ci
    return f"{mean_pct:.{digits}f}±{ci_pct:.{digits}f}"


def fmt_delta(delta: float, digits: int = 2) -> str:
    if pd.isna(delta):
        return "—"
    delta_pct = 100.0 * delta
    sign = "+" if delta_pct >= 0 else ""
    return f"{sign}{delta_pct:.{digits}f}"


def parse_one_csv(path: Path) -> pd.DataFrame:
    m = FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {path.name}")

    meta = m.groupdict()
    df = pd.read_csv(path)

    for k, v in meta.items():
        df[k] = v

    df["K"] = df["K"].astype(int)
    df["N"] = df["N"].astype(int)
    df["lambda_ortho"] = pd.to_numeric(df["lambda_ortho"], errors="coerce")
    df["axis"] = df["axis"].map({"K": "K", "N": "N"})
    df["sweep_value"] = df.apply(lambda r: r["K"] if r["axis"] == "K" else r["N"], axis=1)

    return df


def aggregate_runs(all_df: pd.DataFrame, variants=("cap", "umc")) -> pd.DataFrame:
    all_df = all_df[all_df["variant"].isin(variants)].copy()

    metric_cols = [
        "test_acc",
        "best_val_acc",
        "time_sec",
        "stress_bias_0.0",
        "stress_bias_1.0",
        "stress_bias_2.0",
        "stress_bias_3.0",
        "stress_bias_4.0",
    ]
    metric_cols = [c for c in metric_cols if c in all_df.columns]

    group_cols = [
        "dataset", "train_mode", "axis", "sweep_value", "K", "N",
        "variant", "lambda_ortho"
    ]

    rows = []
    for keys, sub in all_df.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys))
        row["num_seeds"] = sub["seed"].nunique()

        for col in metric_cols:
            row[f"{col}_mean"] = sub[col].mean()
            row[f"{col}_ci95"] = ci95(sub[col])

        # average stress over beta=1..4
        stress_cols = [c for c in ["stress_bias_1.0", "stress_bias_2.0", "stress_bias_3.0", "stress_bias_4.0"] if c in sub.columns]
        if stress_cols:
            per_seed_avg = sub[stress_cols].mean(axis=1)
            row["stress_avg_1to4_mean"] = per_seed_avg.mean()
            row["stress_avg_1to4_ci95"] = ci95(per_seed_avg)

        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(["axis", "sweep_value", "variant"]).reset_index(drop=True)
    return out


def build_markdown_table(summary_df: pd.DataFrame, axis: str) -> str:
    sdf = summary_df[summary_df["axis"] == axis].copy()
    if sdf.empty:
        return f"No {axis}-sweep rows found.\n"

    axis_label = axis

    lines = []
    lines.append(
        f"| {axis_label} | CAP clean | UMC clean | Δ clean | CAP β=4 | UMC β=4 | Δ β=4 |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")

    for sweep_value in sorted(sdf["sweep_value"].unique()):
        sub = sdf[sdf["sweep_value"] == sweep_value]
        cap = sub[sub["variant"] == "cap"]
        umc = sub[sub["variant"] == "umc"]

        if cap.empty or umc.empty:
            continue

        cap = cap.iloc[0]
        umc = umc.iloc[0]

        cap_clean = fmt_pct(cap.get("stress_bias_0.0_mean"), cap.get("stress_bias_0.0_ci95"))
        umc_clean = fmt_pct(umc.get("stress_bias_0.0_mean"), umc.get("stress_bias_0.0_ci95"))
        d_clean = fmt_delta(umc.get("stress_bias_0.0_mean") - cap.get("stress_bias_0.0_mean"))

        cap_b4 = fmt_pct(cap.get("stress_bias_4.0_mean"), cap.get("stress_bias_4.0_ci95"))
        umc_b4 = fmt_pct(umc.get("stress_bias_4.0_mean"), umc.get("stress_bias_4.0_ci95"))
        d_b4 = fmt_delta(umc.get("stress_bias_4.0_mean") - cap.get("stress_bias_4.0_mean"))

        lines.append(
            f"| {int(sweep_value)} | {cap_clean} | {umc_clean} | {d_clean} | {cap_b4} | {umc_b4} | {d_b4} |"
        )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing umc_kn_sweep_*.csv files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to write markdown tables / summary CSVs (default: input_dir)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(input_dir.glob("umc_kn_sweep_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No umc_kn_sweep_*.csv files found in {input_dir}")

    dfs = []
    for p in csv_paths:
        try:
            dfs.append(parse_one_csv(p))
        except Exception as e:
            print(f"[WARN] Skipping {p.name}: {e}")

    if not dfs:
        raise RuntimeError("No valid CSVs parsed.")

    all_df = pd.concat(dfs, ignore_index=True)
    summary_df = aggregate_runs(all_df)

    # Save long summary
    summary_path = output_dir / "summary_long.csv"
    summary_df.to_csv(summary_path, index=False)

    # Build markdown tables
    k_md = build_markdown_table(summary_df, axis="K")
    n_md = build_markdown_table(summary_df, axis="N")

    k_path = output_dir / "k_sweep_table.md"
    n_path = output_dir / "n_sweep_table.md"

    k_path.write_text(k_md)
    n_path.write_text(n_md)

    print("\n=== K sweep table ===\n")
    print(k_md)

    print("\n=== N sweep table ===\n")
    print(n_md)

    print(f"\nWrote:\n- {summary_path}\n- {k_path}\n- {n_path}")


if __name__ == "__main__":
    main()
