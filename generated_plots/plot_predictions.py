import pandas as pd
import matplotlib.pyplot as plt
import re

from matplotlib.pyplot import tight_layout
from scipy.ndimage import gaussian_filter1d  # Import the Gaussian filter

def format_legend_label_fixed(col_name):
    match = re.search(r"topp?=(\d+\.\d+)_topk?=(\d+)_temp=(\d+\.\d+)", col_name)
    if match:
        top_p, top_k, temp = match.groups()
        return rf"$\mathrm{{Top\!-\!p}}$={top_p} $\mathrm{{Top\!-\!k}}$={top_k} $T$={temp}"
    return col_name  # Fallback in case regex fails

# File paths for the two CSVs
accuracy_file = "../data/accuracy_2025-02-19T11_25_42.286-05_00.csv"
coverage_file = "../data/coverage_2025-02-19T10_49_51.690-05_00.csv"

# Load CSV files
df_accuracy = pd.read_csv(accuracy_file)
df_coverage = pd.read_csv(coverage_file)

# Extract steps (assuming both CSVs share the same "Step" values)
steps_acc = df_accuracy["Step"]
steps_cov = df_coverage["Step"]

# Determine metric columns and labels for each subplot
metric_columns_acc = [col for col in df_accuracy.columns if "accuracy" in col and "__" not in col]
metric_columns_cov = [col for col in df_coverage.columns if "coverage" in col and "__" not in col]

# Generate formatted labels
formatted_labels_acc = {col: format_legend_label_fixed(col) for col in metric_columns_acc}
formatted_labels_cov = {col: format_legend_label_fixed(col) for col in metric_columns_cov}

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6))

# Smoothing sigma (adjust as needed)
sigma_value = 1

# Plot accuracy data on the left subplot
for col in metric_columns_acc:
    smoothed_values = gaussian_filter1d(df_accuracy[col].values, sigma=sigma_value)
    ax1.plot(steps_acc, smoothed_values, label=formatted_labels_acc[col], marker='o', linestyle='-')
ax1.set_xlabel("Step", size=14)
ax1.set_ylabel("Accuracy", size=14)
ax1.set_title("Accuracy Progression Over Steps", size=16)
ax1.grid(True)
ax1.set_xticks(steps_acc)
ax1.set_xticklabels([f"{int(x/1000)}k" for x in steps_acc], fontsize=10)

# Plot coverage data on the right subplot
for col in metric_columns_cov:
    smoothed_values = gaussian_filter1d(df_coverage[col].values, sigma=sigma_value)
    ax2.plot(steps_cov, smoothed_values, label=formatted_labels_cov[col], marker='o', linestyle='-')
ax2.set_xlabel("Step", size=14)
ax2.set_ylabel("Coverage", size=14)
ax2.set_title("Coverage Progression Over Steps", size=16)
ax2.grid(True)
ax2.set_xticks(steps_cov)
ax2.set_xticklabels([f"{int(x/1000)}k" for x in steps_cov], fontsize=10)

# Gather handles and labels from both axes for a common legend
handles, labels = ax1.get_legend_handles_labels()


# Adjust layout to make room at the bottom for the legend.
# Increase the bottom margin as needed.
fig.subplots_adjust(bottom=0.3)

# Create a common legend outside the plots at the lower center.
fig.legend(handles, labels, loc="lower center",
           bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize="large")

# Use tight_layout and reserve space at the bottom for the legend.
plt.tight_layout(rect=[0, 0.05, 1, 1])

# Save the combined figure as a high-resolution PNG (300 dpi)
plt.savefig("figures/accuracy_coverage_combined.png", dpi=300, format="png", bbox_inches="tight")

plt.show()
