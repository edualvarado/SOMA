import cv2
import matplotlib.pyplot as plt
import numpy as np

Z_CLIP_MIN = 0.5


def draw_mesh(vertices, faces, camera_pose, intrinsics, frame, return_overlay=True):
    """Draws a 3D mesh onto a video frame.

    This function projects 3D vertices onto a 2D video frame using the specified camera pose and intrinsics,
    and draws the resulting mesh onto the frame. It returns the modified video frame with the mesh overlay.

    Args:
        vertices : An array of 3D vertices of the mesh.
        faces : A list of arrays, each containing indices of vertices forming a mesh face.
        camera_pose : A Pose object for the camera pose.
        intrinsics : An Intrinsic object representing the camera intrinsic matrix and distortion coefficients.
        frame : The video frame onto which the mesh is to be drawn.

    Returns:
        np.ndarray: The video frame with the mesh drawn on it.

    """
    K, d = intrinsics.matrix, intrinsics.distortion
    rvec, tvec = camera_pose.rvec, camera_pose.tvec

    # Trasform the point in the camera space
    R = cv2.Rodrigues(rvec)[0]  # Convert rotation vector to rotation matrix
    vertices_camera = (R @ vertices.T + tvec).T  # Transform vertices to camera space

    # Project 3D points to 2D using the camera parameters
    projected_points = cv2.projectPoints(vertices_camera, np.zeros(3), np.zeros(3), K, d)[0].squeeze()

    # Filter out points that are outside of the frame
    valid_points_mask = (
        (projected_points[:, 0] > 0)
        & (projected_points[:, 0] < frame.shape[1] - 1)
        & (projected_points[:, 1] > 0)
        & (projected_points[:, 1] < frame.shape[0] - 1)
        & (vertices_camera[:, 2] > Z_CLIP_MIN)
    )

    valid_projected_points = projected_points[valid_points_mask]
    valid_vertices_y = vertices[valid_points_mask][:, 1]  # Y-coordinates of valid vertices
    ymin, ymax = np.min(valid_vertices_y), np.max(valid_vertices_y)
    normalized_vertices_y = (valid_vertices_y - ymin) / (ymax - ymin)

    # Create a figure without the frame for plotting
    dpi = 300
    figure = plt.figure(frameon=False, dpi=dpi)
    figure.set_size_inches(frame.shape[1] / dpi, frame.shape[0] / dpi)

    # Fill the figure with the frame
    ax = plt.Axes(figure, [0, 0, 1, 1])
    ax.set_axis_off()
    figure.add_axes(ax)

    # Set transparent background
    ax.patch.set_alpha(0)
    figure.patch.set_alpha(0)

    # Fill the plot with the frame and scatter plot of the projected points
    if return_overlay:
        frame = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.float32)

    ax.imshow(frame[..., ::-1], aspect="equal")
    ax.scatter(valid_projected_points[:, 0], valid_projected_points[:, 1], c=normalized_vertices_y, s=1, alpha=0.4)

    # Plot mesh faces on the image
    for face_indices in faces:
        face_points = projected_points[face_indices]
        if np.all(
            (face_points[:, 0] > 0)
            & (face_points[:, 0] < frame.shape[1] - 1)
            & (face_points[:, 1] > 0)
            & (face_points[:, 1] < frame.shape[0] - 1)
            & (vertices_camera[face_indices][:, 2] > Z_CLIP_MIN)
        ):
            face_points = np.vstack([face_points, face_points[0]])  # Close the loop
            color = (np.mean(vertices[face_indices][:, 1]) - ymin) / (ymax - ymin)
            ax.fill(face_points[:, 0], face_points[:, 1], alpha=0.1, color=plt.cm.viridis(color))

    # Convert the Matplotlib figure to a format that can be displayed on the frame
    figure.canvas.draw()
    image_array = np.array(figure.canvas.renderer.buffer_rgba())
    plt.close(figure)

    # Convert RGBA to BGR for video frame compatibility
    frame_with_mesh = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGRA)

    return frame_with_mesh
