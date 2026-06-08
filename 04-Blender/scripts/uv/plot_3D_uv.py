"""
Plot resulting 3D interpolated points given by Blender script.
"""

import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

import argparse
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from loguru import logger
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

tracking_folder = "inside-humans/uv_detections_charuco-suit/output.json"

with open(tracking_folder, "r") as f:
    data = json.load(f)

# Extract the 3D points (values) from the dictionary
# We assume the values are lists of 3 numbers [x, y, z]
points = []
keypoints = []

# ---

# for key, value in data.items():
#     if isinstance(value, list) and len(value) == 3 and all(isinstance(coord, (int, float)) for coord in value):
#         points.append(value)
#         keypoints.append(key)

for key, value in data.items():
    if isinstance(value, list):
        for point in value:
            if isinstance(point, list) and len(point) == 3 and all(isinstance(coord, (int, float)) for coord in point):
                points.append(point)
                keypoints.append(key)

# Create the scatter plot with conditional coloring
colors = []
sizes = []
for key in keypoints:
    if key == "marker_265_corner_0":  # Replace with your desired key
        colors.append("blue")  # Color for specific key
        sizes.append(3)  # Color for specific key
    else:
        colors.append("blue")  # Default color
        sizes.append(1)  # Color for specific key

# Separate the coordinates into x, y, and z lists
x_coords = [p[0] for p in points]
y_coords = [p[1] for p in points]
z_coords = [p[2] for p in points]

# Create a figure and a 3D subplot
fig = plt.figure(figsize=(10, 8)) # Adjust figure size as needed
ax = fig.add_subplot(111, projection='3d')

# Variables to track axis limits for scaling
# Variables to track axis limits for scaling
x_limits, y_limits, z_limits = [float("inf"), float("-inf")], [float("inf"), float("-inf")], [float("inf"),
                                                                                              float("-inf")]

# Create the scatter plot

ax.scatter(x_coords, y_coords, z_coords, marker='o', s=sizes, c=colors, label='Data Points')

# Update the axis limits
x_limits = [min(x_limits[0], min(x_coords)), max(x_limits[1], max(x_coords))]
y_limits = [min(y_limits[0], min(y_coords)), max(y_limits[1], max(y_coords))]
z_limits = [min(z_limits[0], min(z_coords)), max(z_limits[1], max(z_coords))]

# Set labels for the axes
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')

# Set the title of the plot
ax.set_title('Canonical Model (Static Scanner)')

# Set equal aspect ratio for proper scaling
max_range = max(
    x_limits[1] - x_limits[0],
    y_limits[1] - y_limits[0],
    z_limits[1] - z_limits[0]
) / 2.0
mid_x = (x_limits[0] + x_limits[1]) / 2.0
mid_y = (y_limits[0] + y_limits[1]) / 2.0
mid_z = (z_limits[0] + z_limits[1]) / 2.0

ax.set_box_aspect([1, 1, 1])  # Equal 1:1:1 aspect ratio
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# ax.view_init(elev=30, azim=45)  # Adjust angles as needed

# Optional: Add a legend if needed (useful if plotting multiple datasets)
# ax.legend()

# Add grid for better visualization
ax.grid(True)

# Optional: Set axis limits if needed
# ax.set_xlim([min_x, max_x])
# ax.set_ylim([min_y, max_y])
# ax.set_zlim([min_z, max_z])

# Optional: Ensure equal aspect ratio (might distort plot if ranges differ greatly)
# ax.set_aspect('equal')

# Function to update the view angle for animation
def update(frame):
    ax.view_init(elev=30, azim=frame)

# Create the animation
# ani = FuncAnimation(fig, update, frames=range(0, 360, 1), interval=10)


# Show the plot
plt.show()