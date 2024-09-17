import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import glob
import numpy as np

# Read all client metrics
client_files = glob.glob('client_*_times.csv')
client_dfs = [pd.read_csv(file) for file in client_files]

# Calculate average client metrics
avg_client_metrics = pd.concat(client_dfs).groupby('Round').mean()

# Set the style for seaborn
sns.set(style="whitegrid")

# Create the 3D figure
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

# Define metrics and colors
metrics = ['Train Time', 'Evaluate Time', 'Encryption Time', 'Decryption Time', 'Serialization Time', 'Deserialization Time']
colors = sns.color_palette("husl", len(metrics))

# Parameters for bar width and depth
width = 0.8  # Width of the bars
depth = 0.5  # Depth of the bars

# Plot each metric as bars (without normalization)
for i, (metric, color) in enumerate(zip(metrics, colors)):
    xs = avg_client_metrics.index
    ys = [i] * len(xs)  # Use index of metric for y-axis
    
    zs = avg_client_metrics[metric]
    
    # Use bar3d to plot bars for each metric
    ax.bar3d(xs, ys, np.zeros_like(zs), width, depth, zs, color=color, alpha=0.7)

# Customize the plot
ax.set_xlabel('Round', fontsize=12)
ax.set_zlabel('Time (seconds)', fontsize=12)

# Remove the word "Metrics" from y-axis and improve label size
ax.set_yticks(range(len(metrics)))
ax.set_yticklabels(metrics, fontsize=12)  # Metric labels without "Metrics"

# Set the z-axis to linear scale to make seconds visible
ax.set_zscale('linear')

# Set z-ticks manually to ensure seconds are visible
max_z = avg_client_metrics[metrics].max().max()  # Find the maximum value across all metrics
z_ticks = np.linspace(0, max_z, num=10)  # Create 10 evenly spaced tick marks
ax.set_zticks(z_ticks)
ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))  # Show z values with 1 decimal

# Set a better viewing angle
ax.view_init(elev=20, azim=45)

# Add a title to the plot
plt.title('3D View of Average Client Metrics per Round', fontsize=16)

plt.tight_layout()
plt.savefig('client_metrics_3d_histogram_seconds.png', dpi=300, bbox_inches='tight')
plt.show()


# Server metrics plot (unchanged)
# Server metrics plot in 3D
server_metrics = pd.read_csv('federated_metrics.csv')

# Create the 3D figure
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

# Define metrics for the server and their colors
server_metrics_labels = ['Validation Loss', 'Accuracy']
colors = ['red', 'blue']

# Parameters for bar width and depth
width = 0.8  # Width of the bars
depth = 0.5  # Depth of the bars

# Plot Validation Loss and Accuracy as bars
for i, (metric, color) in enumerate(zip(server_metrics_labels, colors)):
    xs = server_metrics['Round']
    ys = [i] * len(xs)  # y is 0 for Validation Loss, 1 for Accuracy
    
    if metric == 'Validation Loss':
        zs = server_metrics['Average val_loss']
    else:
        zs = server_metrics['Average accuracy']
    
    # Use bar3d to plot bars for each metric
    ax.bar3d(xs, ys, np.zeros_like(zs), width, depth, zs, color=color, alpha=0.7)

# Customize the plot
ax.set_xlabel('Round', fontsize=12)
ax.set_zlabel('Metric Value', fontsize=12)

# Remove the word "Metrics" from y-axis and improve label size
ax.set_yticks([0, 1])
ax.set_yticklabels(server_metrics_labels, fontsize=12)

# Set a linear scale for z-axis
ax.set_zscale('linear')

# Set z-ticks manually to ensure metric values are visible
max_z = max(server_metrics['Average val_loss'].max(), server_metrics['Average accuracy'].max())  # Find max value
z_ticks = np.linspace(0, max_z, num=10)  # Create 10 evenly spaced tick marks
ax.set_zticks(z_ticks)
ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))  # Show z values with 2 decimals

# Set a better viewing angle
ax.view_init(elev=20, azim=45)

# Add a title to the plot
plt.title('3D View of Server Metrics (Validation Loss and Accuracy)', fontsize=16)

plt.tight_layout()
plt.savefig('server_metrics_3d.png', dpi=300, bbox_inches='tight')
plt.show()