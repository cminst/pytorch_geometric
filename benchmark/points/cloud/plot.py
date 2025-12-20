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

LAMBDAS = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
PLOT_DATA = {
    "train_clean": {
        "mean": [82.6, 82.3, 82.4, 83.4, 83.05, 84.72, 84.143, 84.2, 83.45, 81.14],
        "std":  [0.4,  0.6,  0.65, 0.38,  0.57,  0.42,   0.61, 0.73, 0.89,  0.84],
    }
}

def make_umc_lambda_plot(train_mode, out_path):
    """Plot mean±std test accuracy vs lambda_ortho (log x-axis) for UMC,
    for a given train_mode ('train_aug' or 'train_clean').
    """
    if train_mode not in PLOT_DATA:
        raise ValueError(f"Unknown train_mode: {train_mode}")

    mean_list = PLOT_DATA[train_mode]["mean"]
    std_list = PLOT_DATA[train_mode]["std"]
    if len(mean_list) != len(LAMBDAS) or len(std_list) != len(LAMBDAS):
        raise ValueError("Each mean/std list must match the length of LAMBDAS.")

    # Convert to numpy (values are already in %)
    lam = np.array(LAMBDAS, dtype=float)
    mean = np.array(mean_list, dtype=float)
    std = np.array(std_list, dtype=float)

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
    ax.set_title("ModelNet40 Performance vs. $\lambda_{\mathrm{ortho}}$")

    # Custom ticks: show fake-zero tick as "0"
    unique_lam = np.unique(lam)
    real_pos = sorted(v for v in unique_lam if v > 0)
    xticks = [eps] + real_pos
    ax.set_xticks(xticks)
    ax.set_ylim(79.2,86)
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
make_umc_lambda_plot("train_clean","modelnet40_umc_trainclean_lambda_log.pdf")
