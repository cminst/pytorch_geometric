import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogFormatterMathtext

plt.rcParams.update({
    'font.size'        : 14,   # base size for most text
    'axes.titlesize'   : 18,   # title of the axes
    'axes.labelsize'   : 16,   # x‑ and y‑axis labels
    'xtick.labelsize'  : 14,   # tick labels on the x axis
    'ytick.labelsize'  : 14,   # tick labels on the y axis
    'legend.fontsize'  : 14,   # legend text
    'figure.titlesize' : 1,    # figure‑level title (suptitle)
    'font.family'      : "Times New Roman",
    'mathtext.fontset': 'cm'
})

LAMBDAS = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
PLOT_DATA = {
    "train_clean": {
        "mean": [73.53, 73.23, 73.33, 74.33, 73.98, 75.65, 75.07, 75.13, 74.38, 72.07],
        "std":  [1.141, 1.281, 1.316, 1.127, 1.26, 1.155, 1.288, 1.372, 1.484, 1.449],
    }
}

def make_umc_lambda_plot(train_mode, out_path):
    if train_mode not in PLOT_DATA:
        raise ValueError(f"Unknown train_mode: {train_mode}")

    mean_list = PLOT_DATA[train_mode]["mean"]
    std_list = PLOT_DATA[train_mode]["std"]
    if len(mean_list) != len(LAMBDAS) or len(std_list) != len(LAMBDAS):
        raise ValueError("Each mean/std list must match the length of LAMBDAS.")

    lam = np.array(LAMBDAS, dtype=float)
    mean = np.array(mean_list, dtype=float)
    std = np.array(std_list, dtype=float)

    lam_plot = lam.copy()
    pos_mask = lam_plot > 0
    if pos_mask.any():
        min_pos = lam_plot[pos_mask].min()
        eps = min_pos / 10.0
    else:
        eps = 1e-6
    lam_plot[lam_plot == 0] = eps

    # Make the plot
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(lam_plot, mean, "o-")

    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda_{\mathrm{ortho}}$")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("ModelNet40 Performance vs. $\lambda_{\mathrm{ortho}}$", fontweight='bold')

    unique_lam = np.unique(lam)
    real_pos = sorted(v for v in unique_lam if v > 0)
    xticks = [eps] + real_pos
    ax.set_xticks(xticks)
    ax.set_ylim(71,76.5)
    ax.grid(True, alpha=0.3, linestyle="--")
    log_formatter = LogFormatterMathtext(labelOnlyBase=False)
    def tick_label(val, _):
        if np.isclose(val, eps):
            return "0"
        return log_formatter(val)
    ax.xaxis.set_major_formatter(FuncFormatter(tick_label))

    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.15)
    fig.savefig(out_path, dpi=300)
    print(f"Saved {train_mode} plot to: {out_path}")


make_umc_lambda_plot("train_clean","modelnet40_umc_trainclean_lambda_log.pdf")
