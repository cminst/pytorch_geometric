import argparse

import pandas as pd
from rich.console import Console
from rich.table import Table


def _parse_list(arg: str) -> list[str]:
    return [s.strip() for s in arg.split(",") if s.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=str,
        default="umc_ablation_sweep.csv",
        help="Path to the ablation sweep CSV.",
    )
    ap.add_argument(
        "--train_mode",
        type=str,
        default=None,
        help="Optional filter (e.g., train_clean or train_aug).",
    )
    ap.add_argument(
        "--lambda_ortho",
        type=float,
        default=None,
        help="Optional filter for lambda_ortho (e.g., 0).",
    )
    ap.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Optional comma-separated dataset filter.",
    )
    ap.add_argument(
        "--variants",
        type=str,
        default=None,
        help="Optional comma-separated variant filter (e.g., full,no_coords,md_only,deg_only).",
    )
    ap.add_argument(
        "--degree_features",
        type=str,
        default=None,
        help="Optional degree_features filter (e.g., log_deg).",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    if args.train_mode:
        df = df[df["train_mode"] == args.train_mode]
    if args.lambda_ortho is not None:
        df = df[df["lambda_ortho"] == args.lambda_ortho]
    if args.datasets:
        datasets = set(_parse_list(args.datasets))
        df = df[df["dataset"].isin(datasets)]
    if args.variants:
        variants = set(_parse_list(args.variants))
        df = df[df["variant"].isin(variants)]
    if args.degree_features:
        df = df[df["degree_features"] == args.degree_features]

    if df.empty:
        raise SystemExit("No rows match the requested filters.")

    summary = (
        df.groupby(["variant", "dataset"])["test_acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    table = Table(title="UMC Ablation: Mean +/- Std Test Accuracy Per Dataset")
    table.add_column("variant", justify="left")
    table.add_column("dataset", justify="left")
    table.add_column("mean +/- std (%)", justify="right")
    table.add_column("n_seeds", justify="right")

    for _, row in summary.sort_values(["variant", "dataset"]).iterrows():
        mean = row["mean"] * 100.0
        std = 0.0 if pd.isna(row["std"]) else row["std"] * 100.0
        table.add_row(
            row["variant"],
            row["dataset"],
            f"{mean:.2f} +/- {std:.2f}",
            f"{int(row['count'])}",
        )

    Console().print(table)


if __name__ == "__main__":
    main()
