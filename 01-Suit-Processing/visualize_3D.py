import json
import matplotlib.pyplot as plt
import argparse
import matplotlib
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from loguru import logger
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def main():
    parser = argparse.ArgumentParser(description="Triangulate marker positions from multiple cameras")
    parser.add_argument("--folder", type=Path, help="The folder containing the detection data")

    args = parser.parse_args()

    logger.info("================================")
    logger.info("== 4. Visualizing 3D tracking ==")
    logger.info("================================")

    tracking_folder = args.folder / f"tracking_charuco-suit/triangulation/3D-interpolated-N10/triangulation_markers_processed.json"

    with open(tracking_folder, "r") as f:
        data = json.load(f)

    # Access the specific frame (in this case, frame "0")
    frame_data = data["0"]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])  # x:y:z

    # Assign unique colors for each ID
    colors = [
        "r", "g", "b", "c", "m", "y", "k", "orange", "purple", "gray"
    ]  # Extend this list for more unique colors if required
    color_map = {}


    # Variables to track axis limits for scaling
    # Variables to track axis limits for scaling
    x_limits, y_limits, z_limits = [float("inf"), float("-inf")], [float("inf"), float("-inf")], [float("inf"),
                                                                                                  float("-inf")]
    # Iterate through all IDs in the frame and plot their polygons
    for idx, (object_id, coordinates) in enumerate(frame_data.items()):
        # Assign a unique color for the current ID
        color = colors[idx % len(colors)]
        color_map[object_id] = color

        # Swap axes: X -> Z (height), Y -> X, Z -> Y
        swapped_coordinates = [(z, x, y) for x, y, z in coordinates]  # Map new axes
        xs, ys, zs = zip(*swapped_coordinates)  # Extract swapped axes

        # Update the axis limits
        x_limits = [min(x_limits[0], min(xs)), max(x_limits[1], max(xs))]
        y_limits = [min(y_limits[0], min(ys)), max(y_limits[1], max(ys))]
        z_limits = [min(z_limits[0], min(zs)), max(z_limits[1], max(zs))]

        # Create the polygon by connecting the four points
        verts = [swapped_coordinates]  # List of vertices for Poly3DCollection

        # Add the polygon to the plot and fill it with the color
        # poly = Poly3DCollection(verts, color=color, alpha=0.5)  # Adjust alpha for transparency
        # ax.add_collection3d(poly)

        color = cm.viridis(idx / len(frame_data))  # Map index to colormap

        # Scatter the points for better visualization
        ax.scatter(xs, ys, zs, c=color, s=3, label=f"ID {object_id}")

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

    ax.view_init(elev=30, azim=45)  # Adjust angles as needed

    plt.title("3D Coordinates for All IDs (Frame 0)")

    # Add a legend showing each ID and its corresponding color
    # plt.legend()

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()