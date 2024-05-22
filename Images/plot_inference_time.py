import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['resnet50', 'regnet_y_1_6gf', 'regnet_y_3_2gf', 'shufflenet_v2_x1_5', 'resnet34', 'regnet_y_800mf', 'shufflenet_v2_x1_0', 'resnet18', 'regnet_y_400mf', 'shufflenet_v2_x0_5']
representation = ["resnet", "regnet", "regnet", "shufflenet", "resnet", "regnet", "shufflenet", "resnet", "regnet", "shufflenet"]
inference_times = [43.7, 21.7, 25.1, 23.6, 13.6, 14.4, 15.1, 9.8, 14.2, 10.24]  # in milliseconds
mIoU_scores = [60.07, 55.78, 55.69, 59.38, 57.3, 55.64, 58.0, 55.6, 55.0, 53.6]
num_parameters = [128.8, 22.25, 52, 25.1, 28.3, 16.7, 13.2, 18.5, 8.6, 4.3]  # in millions

# Scaling the number of parameters for dot size
sizes = [param * 10 for param in num_parameters]  # arbitrary scaling factor for better visualization

# Mapping representation types to colors
color_map = {"shufflenet": "blue", "regnet": "red", "resnet": "green"}
colors = [color_map[rep] for rep in representation]

# Plotting
fig, ax = plt.subplots()
scatter = ax.scatter(inference_times, mIoU_scores, s=sizes, c=colors, alpha=0.6, edgecolors='w', linewidth=0.5)

# Adding labels
ax.set_xlabel('Inference Time (ms)')
ax.set_ylabel('mIoU')
ax.set_title('LiDAR Semantic Segmentation @ Semantic Kitti Val')

# Adding model names and number of parameters as annotations with dynamic offsets to avoid overlap
offsets = np.linspace(-1, 1, len(models))
for i, (model, param, offset) in enumerate(zip(models, num_parameters, offsets)):
    if i % 2 == 0:
        ax.annotate(f'{model}\n({param}M)', (inference_times[i], mIoU_scores[i] + offset * 0.1), fontsize=8,
                    ha='center', color='black')
    else:
        ax.annotate(f'{model}\n({param}M)', (inference_times[i] + offset * 0.1, mIoU_scores[i]), fontsize=8,
                    ha='center', color='black')

# Set the axes limits
ax.set_xlim(0, 60)
ax.set_ylim(52, 61)

# Define the boundary for the vertical lines
line_boundaries = [25, 50]

# Adding vertical lines at the defined boundaries
for boundary in line_boundaries:
    ax.axvline(x=boundary, color='k', linestyle='--', alpha=0.5)

# Highlighting regions
ax.fill_betweenx([52, 65], 0, line_boundaries[0], color='green', alpha=0.2)
ax.fill_betweenx([52, 65], line_boundaries[0], line_boundaries[1], color='orange', alpha=0.2)
ax.fill_betweenx([52, 65], line_boundaries[1], 60, color='red', alpha=0.2)

plt.grid(True)
plt.show()

