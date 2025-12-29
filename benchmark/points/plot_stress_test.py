import matplotlib.pyplot as plt
import numpy as np

# Data from the stress tests
# UMC + PointWavelet
umc_betas = [1, 2, 3, 4, 5]
umc_accuracies = [91.94, 90.18, 83.25, 73.40, 64.20]

# PointWavelet
pw_betas = [1, 2, 3, 4, 5]
pw_accuracies = [92.35, 88.55, 79.92, 68.92, 59.50]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(umc_betas, umc_accuracies, 'o-', label='UMC + PointWavelet', linewidth=2, markersize=8)
plt.plot(pw_betas, pw_accuracies, 's--', label='PointWavelet', linewidth=2, markersize=8)

# Add labels and title
plt.xlabel('Beta (Sampling Irregularity)', fontsize=12)
plt.ylabel('Stress Test Accuracy (%)', fontsize=12)
plt.title('Stress Test Comparison: UMC + PointWavelet vs PointWavelet', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=12)

# Set x-axis to show integer values
plt.xticks(umc_betas)

# Add a light gray background to highlight the difference
plt.fill_between(umc_betas, umc_accuracies, pw_accuracies, color='gray', alpha=0.1)

# Save the plot
plt.tight_layout()
plt.savefig('umc_pointwavelet_stresstest.pdf', format='pdf', dpi=300, bbox_inches='tight')

print('Plot saved as umc_pointwavelet_stresstest.pdf')
