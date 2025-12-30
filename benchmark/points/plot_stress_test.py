import matplotlib.pyplot as plt
import numpy as np

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

# Data from the stress tests (accuracy in %). Beta=0 omitted per figure spec.
betas = [1, 2, 3, 4, 5]

# PointWavelet (vanilla)
pw_accuracies = [92.35, 88.55, 79.92, 68.92, 59.50]

# PointWavelet + MeanDist fixed quadrature (TODO: fill with beta=1..5 values)
pw_meandist_accuracies = None

# PointWavelet + UMC
umc_accuracies = [91.94, 90.18, 83.25, 73.40, 64.20]

# PointNet++
pnpp_accuracies = [87.34, 83.91, 77.50, 67.6, 56.97]

# DGCNN
dgcnn_accuracies = [88.85, 83.93, 73.12, 60.00, 49.52]

# Create the plot
plt.figure(figsize=(10, 6))
pw_color = "#4C78A8"
umc_color = "#2B6CB0"
pnpp_color = "#7F7F7F"
dgcnn_color = "#404040"
plt.plot(
    betas,
    pw_accuracies,
    "s--",
    label="PointWavelet",
    linewidth=2,
    markersize=8,
    color=pw_color,
)
if pw_meandist_accuracies is not None:
    plt.plot(
        betas,
        pw_meandist_accuracies,
        "D-.",
        label="PointWavelet + MeanDist (fixed)",
        linewidth=2,
        markersize=7,
        color=pw_color,
        alpha=0.7,
    )
plt.plot(
    betas,
    umc_accuracies,
    "o-",
    label="PointWavelet + UMC",
    linewidth=2,
    markersize=8,
    color=umc_color,
)
plt.plot(
    betas,
    pnpp_accuracies,
    "^-",
    label="PointNet++",
    linewidth=2,
    markersize=7,
    color=pnpp_color,
)
plt.plot(
    betas,
    dgcnn_accuracies,
    "v-",
    label="DGCNN",
    linewidth=2,
    markersize=7,
    color=dgcnn_color,
)

# Draw arrows from PointWavelet to PointWavelet + UMC at each beta (skip beta=1).
for beta, pw_acc, umc_acc in zip(betas, pw_accuracies, umc_accuracies):
    if beta == 1:
        continue
    plt.annotate(
        "",
        xy=(beta, umc_acc),
        xytext=(beta, pw_acc),
        arrowprops=dict(arrowstyle="-|>", color="#2ca02c", lw=2.2, alpha=1.0),
    )
    improvement = umc_acc - pw_acc

    plt.annotate(
        f"(+{improvement:.2f})",
        xy=(beta, umc_acc),
        xytext=(8, 8),
        textcoords="offset points",
        color="#2ca02c",
        fontsize=10,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#2ca02c", lw=0.8, alpha=0.9),
    )

# Add labels and title
plt.xlabel(r"$\beta$ (Sampling Irregularity)")
plt.ylabel("Stress Test Accuracy (%)")
plt.title(r"Stress Test Accuracy vs $\beta$", fontweight="bold")
plt.grid(True, alpha=0.3, linestyle="--")
plt.legend()

# Set x-axis to show integer values and give right padding for annotations.
plt.xticks(betas)
plt.xlim(min(betas) - 0.2, max(betas) + 0.4)

# Save the plot
plt.tight_layout()
plt.savefig("umc_pointwavelet_stresstest.pdf", format="pdf", dpi=300, bbox_inches="tight")

print("Plot saved as umc_pointwavelet_stresstest.pdf")
