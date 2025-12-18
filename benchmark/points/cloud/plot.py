import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext

# Global plotting style
plt.rcParams.update({
    "font.size": 14,        # base size for most text
    "axes.titlesize": 16,   # title of the axes
    "axes.labelsize": 14,   # x- and y-axis labels
    "xtick.labelsize": 12,  # tick labels on the x axis
    "ytick.labelsize": 12,  # tick labels on the y axis
    "legend.fontsize": 12,  # legend text
    "figure.titlesize": 2   # figure-level title (suptitle)
})

# Path to your CSV
csv_path = "modelnet10_umc_sweep.csv"  # adjust if needed
df = pd.read_csv(csv_path)

def make_umc_lambda_plot(df, train_mode, out_path):
    """
    Plot mean±std test accuracy vs lambda_ortho (log x-axis) for UMC,
    for a given train_mode ('train_aug' or 'train_clean').
    """
    # Filter to this training regime and UMC runs
    sub = df[(df["train_mode"] == train_mode) & (df["variant"] == "umc")].copy()
    sub["lambda_ortho"] = pd.to_numeric(sub["lambda_ortho"])

    # Aggregate accuracy over seeds for each lambda_ortho
    agg = (
        sub.groupby("lambda_ortho")["test_acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("lambda_ortho")
    )

    # Convert to numpy and to %
    lam = agg["lambda_ortho"].to_numpy()
    mean = (agg["mean"].to_numpy() * 100.0)  # accuracy in %
    std = (agg["std"].to_numpy() * 100.0)

    # Fake a tiny positive value for lambda=0 so log scale works
    lam_plot = lam.copy()
    pos_mask = lam_plot > 0
    if pos_mask.any():
        min_pos = lam_plot[pos_mask].min()
        eps = min_pos / 10.0
    else:
        eps = 1e-6  # fallback, shouldn't happen here
    lam_plot[lam_plot == 0] = eps

    # Make the plot
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.errorbar(lam_plot, mean, yerr=std, fmt="o-", capsize=4)

    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda_{\mathrm{ortho}}$")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("ModelNet10 Performance vs. $\lambda_{\mathrm{ortho}}$")
    ax.set_ylim(80, 88)

    # Custom ticks: show fake-zero tick as "0"
    unique_lam = np.unique(lam)
    real_pos = sorted(v for v in unique_lam if v > 0)
    xticks = [eps] + real_pos
    ax.set_xticks(xticks)
    log_formatter = LogFormatterMathtext(labelOnlyBase=False)
    def tick_label(val, _):
        if np.isclose(val, eps):
            return "0"
        return log_formatter(val)
    ax.xaxis.set_major_formatter(FuncFormatter(tick_label))

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"Saved {train_mode} plot to: {out_path}")


# Make both plots
make_umc_lambda_plot(df, "train_aug",  "modelnet10_umc_trainaug_lambda_log.pdf")
make_umc_lambda_plot(df, "train_clean","modelnet10_umc_trainclean_lambda_log.pdf")
