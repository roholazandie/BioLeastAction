# import pandas as pd
# import matplotlib.pyplot as plt
# import re
# import numpy as np
# from matplotlib.pyplot import tight_layout
# from scipy.ndimage import gaussian_filter1d  # Import the Gaussian filter
#
# def format_legend_label_fixed(col_name):
#     match = re.search(r"topp?=(\d+\.\d+)_topk?=(\d+)_temp=(\d+\.\d+)", col_name)
#     if match:
#         top_p, top_k, temp = match.groups()
#         return rf"$\mathrm{{Top\!-\!p}}$={top_p} $\mathrm{{Top\!-\!k}}$={top_k} $T$={temp}"
#     return col_name  # Fallback in case regex fails
#
# # File paths for the two CSVs
# # accuracy_file = "../data/accuracy_2025-02-19T11_25_42.286-05_00.csv"
# # coverage_file = "../data/coverage_2025-02-19T10_49_51.690-05_00.csv"
#
# accuracy_file = "../data/accuracy_2025-03-05T13_29_53.381-05_00.csv"
# coverage_file = "../data/coverage_2025-02-19T10_49_51.690-05_00.csv"
#
# # Load CSV files
# df_accuracy = pd.read_csv(accuracy_file)
# df_coverage = pd.read_csv(coverage_file)
#
# # Extract steps (assuming both CSVs share the same "Step" values)
# steps_acc = df_accuracy["Step"]
# steps_cov = df_coverage["Step"]
#
# # Determine metric columns for each subplot
# metric_columns_acc = [col for col in df_accuracy.columns if "accuracy" in col and "__" not in col]
# metric_columns_cov = [col for col in df_coverage.columns if "coverage" in col and "__" not in col]
#
# # Generate formatted labels (used in the progression plots)
# formatted_labels_acc = {col: format_legend_label_fixed(col) for col in metric_columns_acc}
# formatted_labels_cov = {col: format_legend_label_fixed(col) for col in metric_columns_cov}
#
# # Smoothing sigma (adjust as needed)
# sigma_value = 1
#
# ###############################
# # Figure 1: Accuracy & Coverage Progression Over Steps
# ###############################
#
# # Create a figure with two subplots side by side for progression plots
# fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6))
#
# # Plot accuracy data on the left subplot
# for col in metric_columns_acc:
#     smoothed_values = gaussian_filter1d(df_accuracy[col].values, sigma=sigma_value)
#     ax1.plot(steps_acc, smoothed_values, label=formatted_labels_acc[col], marker='o', linestyle='-')
# ax1.set_xlabel("Step", size=14)
# ax1.set_ylabel("Accuracy", size=14)
# ax1.set_title("Accuracy Progression Over Steps", size=16)
# ax1.grid(True)
# ax1.set_xticks(steps_acc)
# ax1.set_xticklabels([f"{int(x/1000)}k" for x in steps_acc], fontsize=10)
#
# # Plot coverage data on the right subplot
# for col in metric_columns_cov:
#     smoothed_values = gaussian_filter1d(df_coverage[col].values, sigma=sigma_value)
#     ax2.plot(steps_cov, smoothed_values, label=formatted_labels_cov[col], marker='o', linestyle='-')
# ax2.set_xlabel("Step", size=14)
# ax2.set_ylabel("Coverage", size=14)
# ax2.set_title("Coverage Progression Over Steps", size=16)
# ax2.grid(True)
# ax2.set_xticks(steps_cov)
# ax2.set_xticklabels([f"{int(x/1000)}k" for x in steps_cov], fontsize=10)
#
# # Gather handles and labels from the accuracy plot for a common legend.
# handles, labels = ax1.get_legend_handles_labels()
#
# # Adjust layout to make room at the bottom for the legend.
# fig1.subplots_adjust(bottom=0.3)
# fig1.legend(handles, labels, loc="lower center",
#             bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize="large")
#
# plt.tight_layout(rect=[0, 0.05, 1, 1])
# # Save the progression figure
# plt.savefig("figures/accuracy_coverage_progression.png", dpi=300, format="png", bbox_inches="tight")
# plt.show()
#
# ###############################
# # Figure 2: Average Accuracy vs Average Coverage with Std Shading
# ###############################
#
# # Compute smoothed values for each metric and store in lists
# smoothed_acc_list = []
# for col in metric_columns_acc:
#     smoothed_acc = gaussian_filter1d(df_accuracy[col].values, sigma=sigma_value)
#     smoothed_acc_list.append(smoothed_acc)
# smoothed_cov_list = []
# for col in metric_columns_cov:
#     smoothed_cov = gaussian_filter1d(df_coverage[col].values, sigma=sigma_value)
#     smoothed_cov_list.append(smoothed_cov)
#
# # Compute average and std across all metrics for each step
# avg_smoothed_acc = np.mean(smoothed_acc_list, axis=0)
# std_smoothed_acc = np.std(smoothed_acc_list, axis=0)
# avg_smoothed_cov = np.mean(smoothed_cov_list, axis=0)
#
# # Create a new figure for the average accuracy vs coverage plot with std shading
# fig2, ax = plt.subplots(figsize=(8, 6))
# ax.plot(avg_smoothed_cov, avg_smoothed_acc, marker='o', linestyle='-', label="Average Accuracy")
# ax.fill_between(avg_smoothed_cov,
#                 avg_smoothed_acc - std_smoothed_acc,
#                 avg_smoothed_acc + std_smoothed_acc,
#                 alpha=0.3, label="Std Deviation")
# ax.set_xlabel("Average Coverage", size=14)
# ax.set_ylabel("Average Accuracy", size=14)
# ax.set_title("Average Accuracy vs Average Coverage", size=16)
# ax.grid(True)
# ax.legend(fontsize="large")
#
# plt.tight_layout()
# # Save the average accuracy vs coverage figure with shading separately
# plt.savefig("figures/avg_accuracy_vs_coverage_with_std.png", dpi=300, format="png", bbox_inches="tight")
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from matplotlib.pyplot import tight_layout
from scipy.ndimage import gaussian_filter1d  # Import the Gaussian filter

def format_legend_label_fixed(col_name):
    match = re.search(r"topp?=(\d+\.\d+)_topk?=(\d+)_temp=(\d+\.\d+)", col_name)
    if match:
        top_p, top_k, temp = match.groups()
        return rf"$\mathrm{{Top\!-\!p}}$={top_p} $\mathrm{{Top\!-\!k}}$={top_k} $T$={temp}"
    return col_name  # Fallback in case regex fails

# File paths for the two CSVs
accuracy_file = "../data/accuracy_2025-03-05T13_29_53.381-05_00.csv"
coverage_file = "../data/coverage_2025-02-19T10_49_51.690-05_00.csv"

# Load CSV files
df_accuracy = pd.read_csv(accuracy_file)
df_coverage = pd.read_csv(coverage_file)

# Extract steps (assuming both CSVs share the same "Step" values)
steps_acc = df_accuracy["Step"]
steps_cov = df_coverage["Step"]

# Determine metric columns for each subplot
metric_columns_acc = [col for col in df_accuracy.columns if "accuracy" in col and "__" not in col]
metric_columns_cov = [col for col in df_coverage.columns if "coverage" in col and "__" not in col]

# Generate formatted labels (used in the progression plots)
formatted_labels_acc = {col: format_legend_label_fixed(col) for col in metric_columns_acc}
formatted_labels_cov = {col: format_legend_label_fixed(col) for col in metric_columns_cov}

# Smoothing sigma (adjust as needed)
sigma_value = 1

###############################
# Figure 1: Accuracy & Coverage Progression Over Steps
###############################

# Create a figure with two subplots side by side for progression plots
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6))

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

# Gather handles and labels from the accuracy plot for a common legend.
handles, labels = ax1.get_legend_handles_labels()

# Adjust layout to make room at the bottom for the legend.
fig1.subplots_adjust(bottom=0.3)
fig1.legend(handles, labels, loc="lower center",
            bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize="large")

plt.tight_layout(rect=[0, 0.05, 1, 1])
# Save the progression figure
plt.savefig("../figures/accuracy_coverage/accuracy_coverage_progression.png", dpi=300, format="png", bbox_inches="tight")
plt.show()

###############################
# Figure 2: Average Accuracy vs Average Coverage with Std Shading
###############################

# Compute smoothed values for each metric and store in lists
smoothed_acc_list = []
for col in metric_columns_acc:
    smoothed_acc = gaussian_filter1d(df_accuracy[col].values, sigma=sigma_value)
    smoothed_acc_list.append(smoothed_acc)
smoothed_cov_list = []
for col in metric_columns_cov:
    smoothed_cov = gaussian_filter1d(df_coverage[col].values, sigma=sigma_value)
    smoothed_cov_list.append(smoothed_cov)

# Compute average and std across all metrics for each step
avg_smoothed_acc = np.mean(smoothed_acc_list, axis=0)
std_smoothed_acc = np.std(smoothed_acc_list, axis=0)
avg_smoothed_cov = np.mean(smoothed_cov_list, axis=0)

# In some cases, the same coverage value corresponds to different accuracy values.
# We will group by rounded coverage (to 5 decimal places) and for each unique coverage,
# choose the maximum accuracy (and its std) among the ones available.
cov_to_best = {}
for i, cov in enumerate(avg_smoothed_cov):
    cov_rounded = round(cov, 5)
    acc = avg_smoothed_acc[i]
    std_val = std_smoothed_acc[i]
    if cov_rounded not in cov_to_best or acc > cov_to_best[cov_rounded][0]:
        cov_to_best[cov_rounded] = (acc, std_val)

# Create new arrays sorted by coverage
unique_cov_sorted = np.array(sorted(cov_to_best.keys()))
max_acc = np.array([cov_to_best[cov][0] for cov in unique_cov_sorted])
max_std = np.array([cov_to_best[cov][1] for cov in unique_cov_sorted])

# Create a new figure for the average accuracy vs coverage plot with std shading
fig2, ax = plt.subplots(figsize=(8, 6))
ax.plot(unique_cov_sorted, max_acc, marker='o', linestyle='-')
ax.fill_between(unique_cov_sorted,
                max_acc - max_std,
                max_acc + max_std,
                alpha=0.3)
ax.set_xlabel("Coverage", size=14)
ax.set_ylabel("Accuracy", size=14)
ax.set_title("Accuracy vs Coverage", size=16)
ax.grid(True)

plt.tight_layout()
# Save the average accuracy vs coverage figure with shading separately
plt.savefig("../figures/accuracy_coverage/avg_accuracy_vs_coverage_with_std.png", dpi=300, format="png", bbox_inches="tight")
plt.show()

