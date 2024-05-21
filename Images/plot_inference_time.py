import matplotlib.pyplot as plt

# Data
models = ['resnet50', 'regnet_y_1_6gf','regnet_y_3_2gf','shufflenet_v2_x1_5','resnet34','regnet_y_800mf','shufflenet_v2_x1_0']
representation = ["resnet", "regnet", "regnet","shufflenet","resnet", "regnet","shufflenet"]
inference_times = [43.7, 21.7, 25.1, 23.6, 13.6, 14.4, 15.1]  # in milliseconds
mIoU_scores = [60.07, 55.78, 55.69, 59.38, 57.3, 55.64, 60.0]
num_parameters = [128.8, 22.25, 52, 25.1, 28.3, 16.7, 13.2]  # in millions

# Scaling the number of parameters for dot size
sizes = [param * 10 for param in num_parameters]  # arbitrary scaling factor for better visualization

# Mapping representation types to colors
color_map = {"shufflenet": "blue", "regnet": "red", "resnet": "green"}
colors = [color_map[rep] for rep in representation]

# Plotting
fig, ax = plt.subplots()
scatter = ax.scatter(inference_times, mIoU_scores, s=sizes, c=colors)

# Adding labels
ax.set_xlabel('Inference Time (ms)')
ax.set_ylabel('mIoU')
ax.set_title('LiDAR Semantic Segmentation @ Semantic Kitti Val')

# Adding model names as annotations
for i, model in enumerate(models):
    ax.annotate(model, (inference_times[i], mIoU_scores[i]))

# Set the axes limits
ax.set_xlim(0, 60)
ax.set_ylim(52, 65)

# Adding legend
#legend1 = ax.legend(*scatter.legend_elements(),
#                    loc="lower right", title="Representation")
#ax.add_artist(legend1)

# Define the boundary for the vertical line
line_boundary = 25

# Adding vertical line at the defined boundary
ax.axvline(x=line_boundary, color='k', linestyle='--', alpha=0.5)
ax.axvline(x=50, color='k', linestyle='--', alpha=0.5)

# Highlighting left side of the line with green color
ax.fill_betweenx([20, 80], 0, line_boundary, color='green', alpha=0.2)

# Highlighting right side of the line with red color
ax.fill_betweenx([20, 80], line_boundary, 50, color='orange', alpha=0.2)
ax.fill_betweenx([20, 80], 50, 100, color='red', alpha=0.2)

plt.grid(True)
plt.show()
