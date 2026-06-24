import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix data
conf_matrix = np.array([[90.2, 9.8, 0.0], 
                        [6.7, 93.3, 0.0], 
                        [0.0, 0.0, 100.0]])

case4_matrix = np.array([[82.8, 13.8, 3.4], 
                          [13.2, 85.3, 1.5], 
                          [12.0, 4.0, 84.0]])

exonly_matrix = np.array([[90.2, 9.8, 0.0], 
                          [6.9, 93.1, 0.0], 
                          [42.5, 0.0, 57.5]])

weightingNN_matrix = np.array([[78.2, 1.0, 20.8], 
                               [18.8, 73.6, 7.6], 
                               [0.2, 0.0, 99.8]])

# Labels with larger font size
labels = ["Crossing", "Left Turn", "Right Turn"]

# Create the figure and axes (2 rows × 2 columns) with reduced vertical spacing
fig, axes = plt.subplots(2, 2, figsize=(12, 12), gridspec_kw={'wspace': 0.3, 'hspace': 0.15})

# Define a consistent color range for all plots
vmin, vmax = 0, 100

# Plot Case 1 (Top Left) - Remove xticklabels
sns.heatmap(conf_matrix, annot=True, fmt=".1f", cmap="Blues", xticklabels=False, yticklabels=labels,
            annot_kws={"size": 26}, ax=axes[0, 0], vmin=vmin, vmax=vmax, cbar=False)
axes[0, 0].set_title("Proposed model", fontsize=22, fontweight='bold')

# Plot Case 2 (Top Right) - Remove xticklabels & yticklabels
sns.heatmap(case4_matrix, annot=True, fmt=".1f", cmap="Blues", xticklabels=False, yticklabels=False,
            annot_kws={"size": 26}, ax=axes[0, 1], vmin=vmin, vmax=vmax, cbar=False)
axes[0, 1].set_title("Model from previous research", fontsize=22, fontweight='bold')

# Plot Case 3 (Bottom Left)
sns.heatmap(exonly_matrix, annot=True, fmt=".1f", cmap="Blues", xticklabels=labels, yticklabels=labels,
            annot_kws={"size": 26}, ax=axes[1, 0], vmin=vmin, vmax=vmax, cbar=False)
axes[1, 0].set_title("Ablation study 1", fontsize=22, fontweight='bold')

# Plot Case 4 (Bottom Right) - Remove yticklabels, keep xticklabels
cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.78])  # Color bar positioning

sns.heatmap(weightingNN_matrix, annot=True, fmt=".1f", cmap="Blues", xticklabels=labels, yticklabels=False,
            annot_kws={"size": 26}, ax=axes[1, 1], vmin=vmin, vmax=vmax, cbar=True, cbar_ax=cbar_ax)
axes[1, 1].set_title("Ablation study 2", fontsize=22, fontweight='bold')

# Increase font size for tick labels **only where they exist**
for ax in [axes[1, 0], axes[1, 1]]:  # Only bottom row should have xticklabels
    ax.set_xticklabels(labels, fontsize=17, rotation=0)

for ax in [axes[0, 0], axes[1, 0]]:  # Only left column should have yticklabels
    ax.set_yticklabels(labels, fontsize=17, rotation=90)

# Remove redundant labels
for ax in [axes[0, 0], axes[0, 1]]:  # Top row -> No xlabel
    ax.set_xlabel("")

for ax in [axes[0, 1], axes[1, 1]]:  # Right column -> No ylabel
    ax.set_ylabel("")

# Global Labels
fig.text(0.08, 0.5, "Ground Truth Label", va='center', ha='center', fontsize=22, fontweight='bold', rotation=90)
fig.text(0.5, 0.06, "Predicted Label", va='center', ha='center', fontsize=22, fontweight='bold')

# Adjust color bar label
cbar_ax.set_ylabel("Accuracy [%]", fontsize=22, fontweight='bold')
cbar_ax.tick_params(labelsize=20)

# Save the figure
plt.savefig("confusion_matrices.png", dpi=300, bbox_inches='tight')  # Save as PNG

# Show the plot
# plt.show()



# Confusion matrix data
EX_conf_matrix = np.array([[100.0, 0.0, 0.0], 
                           [0.0, 100.0, 0.0],
                           [2.4, 1.1, 96.5]])

IM_conf_matrix = np.array([[73.3, 2.8, 12.6, 11.3], 
                          [8.5, 25.9, 0.0, 65.6], 
                          [0.0, 0.0, 98.0, 2.0],
                          [3.3, 6.2, 15.7, 74.8]])



# Labels with larger font size
EX_labels = ["Left Hand", "Right Hand", "Neutral"]
IM_labels = ["Left \nOvershoulder\n\n", "Left \nLook\n\n", "Right \nLook\n\n", "Neutral\n\n"]

# Create the figure and axes (2 rows × 2 columns) with reduced vertical spacing
fig_gestures, axes_gestures = plt.subplots(1, 2, figsize=(24, 12), gridspec_kw={'wspace': 0.3, 'hspace': 0.15})
fig_gestures.subplots_adjust(left=0.12, right=0.88, bottom=0.15, top=0.9, wspace=0.3)

# Define a consistent color range for all plots
vmin, vmax = 0, 100


sns.heatmap(EX_conf_matrix, annot=True, fmt=".1f", cmap="Blues", xticklabels=EX_labels, yticklabels=EX_labels,
            annot_kws={"size": 48}, ax=axes_gestures[0], vmin=vmin, vmax=vmax, cbar=False)
axes_gestures[0].set_title("Explicit Gestures", fontsize=40, fontweight='bold')
axes_gestures[0].set_xticklabels(EX_labels, fontsize=30, rotation=0)
axes_gestures[0].set_yticklabels(EX_labels, fontsize=30, rotation=90)


cbar_ax_gestures = fig_gestures.add_axes([0.92, 0.1, 0.02, 0.8])  # Color bar positioning
sns.heatmap(IM_conf_matrix, annot=True, fmt=".1f", cmap="Blues", xticklabels=IM_labels, yticklabels=IM_labels,
            annot_kws={"size": 48}, ax=axes_gestures[1], vmin=vmin, vmax=vmax, cbar=True, cbar_ax=cbar_ax_gestures)
axes_gestures[1].set_title("Implicit Gestures", fontsize=40, fontweight='bold')
axes_gestures[1].set_xticklabels(IM_labels, fontsize=30, rotation=0)
axes_gestures[1].set_yticklabels(IM_labels, fontsize=30, rotation=90, ha='center')




# Global Labels
fig_gestures.text(0.08, 0.5, "Ground Truth Label", va='center', ha='center', fontsize=40, fontweight='bold', rotation=90)
fig_gestures.text(0.5, 0.03, "Predicted Label", va='center', ha='center', fontsize=40, fontweight='bold')

# Adjust color bar label
# 设置 colorbar 刻度为 0, 20, 40, 60, 80, 100
cbar_ax_gestures.set_yticks([0, 20, 40, 60, 80, 100])
cbar_ax_gestures.set_yticklabels(["0", "20", "40", "60", "80", "100"])
cbar_ax_gestures.set_ylabel("Accuracy [%]", fontsize=40, fontweight='bold')
cbar_ax_gestures.tick_params(labelsize=40)

# Save the figure
plt.savefig("Gestures_confusion_matrices.png", dpi=300, bbox_inches='tight')  # Save as PNG

# Show the plot
# plt.show()
