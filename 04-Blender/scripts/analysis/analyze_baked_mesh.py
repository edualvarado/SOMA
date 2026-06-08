"""
Script: analyze_baked_mesh.py

shot_001 - 120 - 180 (squat with no weight) 160 full down --- 100 200 (160
shot_012 - 130 - 220 (squat with heavy weight) 185 full down --- 130 220 (180)
shot_012 - 1950 - 2025 (squat with heavy weight) 1990 full down

squats: shots 1, 4, 8, 12
biceps: shots 2, 6, 10, 14
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import glob
import shutil
import trimesh
from matplotlib.ticker import FuncFormatter
from matplotlib.collections import PolyCollection
from matplotlib import colormaps
import matplotlib.colors as colors

# --- 1. USER CONFIGURATION ---

# -- Data Source --
# shot = "shot_004"
SHOT_LIST = ["shot_001"]
# SHOT_LIST = ["shot_006", "shot_010", "shot_014"]

# -- Analysis Mode --
# 'displacement': Fast approximation of deformation (removes translation only).
# 'pure_deformation':  Most accurate deformation (removes translation and rotation).
# 'stretch_change': Dig more
# 'debug_motion'
ANALYSIS_MODE = 'pure_deformation'

# -- Frame & Video Settings --
START_FRAME = 1
FRAME_LIMIT = 300
VIDEO_FPS = 24
CLEANUP_PNGS = False  # Deletes individual frame images after creating the video.

# -- Visualization & Normalization --
# Define the "window of interest" for the metric's values.
# Values below the lower bound will be the darkest color, above the upper will be the brightest.
if ANALYSIS_MODE == 'stretch_change':
    # For diverging colormaps, we set the max negative and positive values.
    # Values between these will use the full color gradient.
    LOWER_THRESHOLD = -2.5 # Represents negative change in stretch
    UPPER_THRESHOLD = 2.5  # Represents positive change in stretch
else: # For deformation modes
    LOWER_THRESHOLD = 2.0
    UPPER_THRESHOLD = 20.0

# --- 3. HELPER FUNCTIONS ---

def load_obj(filepath):
    """Loads vertices and faces from a .obj file."""
    V, F = [], []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                V.append([float(x) for x in values[1:4]])
            elif values[0] == 'f':
                F.append([int(x.split('/')[0]) - 1 for x in values[1:4]])
    return np.array(V), np.array(F)

def format_as_percentage(x, pos):
    """Converts a raw deformation value to a percentage string for the color bar."""
    return f"{x * 100:.2f}%"

def format_as_mm(x, pos):
    """Formats a value for the color bar legend in millimeters."""
    # Show one decimal place for values >= 1.0, otherwise show two.
    if x >= 1.0:
        return f"{x:.1f} mm"
    else:
        return f"{x:.2f} mm"

# Change the function name and the output string from "mm" to "m"
def format_as_meters(x, pos):
    """Formats a value to six decimal places with 'm'."""
    return f"{x:.6f} m"

def create_video(image_folder, video_path, fps, cleanup):
    """Compiles a video from a folder of images."""
    print("\nCreating video...")
    images = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    if not images:
        print("No frames were generated to create a video.")
        return
    frame = cv2.imread(images[0]);
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v');
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    for image in images: video.write(cv2.imread(image))
    video.release();
    print(f"Video saved successfully to {video_path}")
    if cleanup:
        print("Cleaning up intermediate PNG files...");
        shutil.rmtree(image_folder);
        print("Cleanup complete.")

def compute_stretch(V, F):
    """Calculates the local stretch for each vertex."""
    stretch = np.zeros(len(V))
    vertex_neighbors = {i: set() for i in range(len(V))}
    for face in F:
        for i in range(3):
            vertex_neighbors[face[i]].update([face[(i + 1) % 3], face[(i + 2) % 3]])
    for i, neighbors in vertex_neighbors.items():
        if not neighbors: continue
        neighbor_positions = np.array([V[j] for j in neighbors])
        stretch[i] = np.linalg.norm(neighbor_positions - V[i], axis=1).mean()

    return stretch

def calculate_triangle_areas(vertices, faces):
    """Calculates the area of each triangle in a mesh."""
    # Get vertices for each face
    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]

    # Calculate edge vectors
    edge1 = p1 - p0
    edge2 = p2 - p0

    # Area is half the magnitude of the cross product
    cross_product = np.cross(edge1, edge2)
    areas = 0.5 * np.linalg.norm(cross_product, axis=1)

    return areas

def frustum(left, right, bottom, top, znear, zfar):
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = +2.0 * znear / (right - left)
    M[1, 1] = +2.0 * znear / (top - bottom)
    M[2, 2] = -(zfar + znear) / (zfar - znear)
    M[0, 2] = (right + left) / (right - left)
    M[1, 2] = (top + bottom) / (top - bottom)
    M[2, 3] = -2.0 * znear * zfar / (zfar - znear)
    M[3, 2] = -1.0
    return M

def perspective(fovy, aspect, znear, zfar):
    h = np.tan(0.5 * np.radians(fovy)) * znear
    w = h * aspect
    return frustum(-w, w, -h, h, znear, zfar)

def translate(x, y, z):
    return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], dtype=float)

def xrotate(theta):
    t = np.pi * theta / 180;
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=float)

def yrotate(theta):
    t = np.pi * theta / 180;
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], dtype=float)

def zrotate(theta):
    t = np.pi * theta / 180;
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)

def compute_signed_distance(V_ref_norm, F_ref, V_aligned):
    """Computes signed distances by projecting deformation onto reference normals."""
    mesh_ref = trimesh.Trimesh(vertices=V_ref_norm, faces=F_ref, process=False)
    vertex_normals = mesh_ref.vertex_normals
    displacement_vectors = V_aligned - V_ref_norm
    # Use einsum for an efficient row-wise dot product
    signed_distances = np.einsum('ij,ij->i', displacement_vectors, vertex_normals)
    return signed_distances

def create_shot(shot):
    # --- 2. PATHS AND SETUP ---

    # -- Input/Output Paths --
    BASE_DIR = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans"
    BAKED_MESHES_DIR = os.path.join(BASE_DIR, "output", shot)
    BAKED_INITIAL_MESH_DIR = os.path.join(BASE_DIR, "output/shot_000/")
    OUTPUT_DIRECTORY = os.path.join(BASE_DIR, "analysis", shot)
    OUTPUT_PATH = os.path.join(OUTPUT_DIRECTORY, f"{shot}_{ANALYSIS_MODE}_analysis.mp4")

    # Make sure output directory exists and create a temporary folder for frames
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    temp_frame_dir = os.path.join(OUTPUT_DIRECTORY, "temp_frames")
    if os.path.exists(temp_frame_dir): shutil.rmtree(temp_frame_dir)
    os.makedirs(temp_frame_dir)

    # --- 4. MAIN PROCESSING LOGIC ---

    print(f"--- Running '{ANALYSIS_MODE.upper()}' analysis for shot: {shot} ---")

    # Phase A: Setup (Before the Loop)

    # -- Consistent Normalization Setup --

    all_obj_files = sorted([f for f in os.listdir(BAKED_MESHES_DIR) if f.endswith('.obj')])

    # ---------
    # Option 1: Take the initial one of the sequence
    # reference_file_path = os.path.join(BAKED_MESHES_DIR, all_obj_files[0])

    # Option 2: Take the initial one of the workout
    only_file = sorted([f for f in os.listdir(BAKED_INITIAL_MESH_DIR) if f.endswith('.obj')])
    reference_file_path = os.path.join(BAKED_INITIAL_MESH_DIR, only_file[0])
    # ---------

    V_ref, F_ref = load_obj(reference_file_path)

    # Calculate normalization parameters ONCE from the reference mesh
    offset = (V_ref.max(0) + V_ref.min(0)) / 2  # Center point of bounding box
    scale = max(V_ref.max(0) - V_ref.min(0))  # Longest side
    V_ref_norm = (V_ref - offset) / scale  # Substracting offset to move to origin and with largest side equal to 1.0
    stretch_ref = compute_stretch(V_ref_norm, F_ref)

    # View Projection Setup
    model = xrotate(-90) @ yrotate(0) @ zrotate(-45);
    view = translate(0, 0, -2.5);
    proj = perspective(25, 1, 1, 100)
    MVP = proj @ view @ model

    # Phase B: Frame Processing Loop

    files_to_process = all_obj_files[START_FRAME:]
    if FRAME_LIMIT is not None:
        files_to_process = files_to_process[:FRAME_LIMIT]

    if ANALYSIS_MODE == 'debug_motion':
        # This mode prints a detailed breakdown to the console in millimeters (mm).

        macro_disps_mm = []
        micro_procrustes_disps_mm = []
        stretch_disps_mm = []

        print("\nProcessing frames to measure deformation in millimeters (mm)...")
        for i, obj_file in enumerate(files_to_process):
            V, F = load_obj(os.path.join(BAKED_MESHES_DIR, obj_file))
            V_norm = (V - offset) / scale

            # --- Metric Calculations (in normalized units) ---
            cm_ref = V_ref_norm.mean(axis=0)
            cm_current = V_norm.mean(axis=0)
            macro_disp_normalized = np.linalg.norm(cm_current - cm_ref)

            _, V_aligned, _ = trimesh.registration.procrustes(V_norm, V_ref_norm)
            micro_disp_procrustes_normalized = np.linalg.norm(V_aligned - V_ref_norm, axis=1).mean()

            # Calculates the average distance to neighboring vertices.
            stretch_current = compute_stretch(V_norm, F)

            # The metric is the DIFFERENCE from the reference stretch
            stretch_normalized = stretch_current - stretch_ref

            # --- Convert to Millimeters ---
            # 1. Convert to meters by multiplying by the original scale factor.
            macro_disp_m = macro_disp_normalized * scale
            micro_disp_procrustes_m = micro_disp_procrustes_normalized * scale
            stretch_m = stretch_normalized * scale
            # 2. Convert meters to millimeters.
            macro_disp_mm = macro_disp_m * 1000
            micro_disp_procrustes_mm = micro_disp_procrustes_m * 1000
            stretch_mm = stretch_m * 1000

            macro_disps_mm.append(macro_disp_mm)
            micro_procrustes_disps_mm.append(micro_disp_procrustes_mm)
            stretch_disps_mm.append(stretch_mm)

            if i % 10 == 0:
                # CHANGED: Displaying mm with 3 decimal places
                print(f"Frame {START_FRAME + i}: Pure Deformation = {micro_disp_procrustes_mm:.3f} mm")

        print("\n" + "=" * 42)
        # CHANGED: Title is now '(in mm)'
        print("---      FINAL DEBUG SUMMARY (in mm)     ---")
        print("=" * 42)
        print(f"Analysis for shot: {shot}\n")
        print("AVERAGE DEFORMATION ACROSS ALL FRAMES (mm):")
        print(f"  Macro Motion (Center Shift): {np.mean(macro_disps_mm):.3f} mm")
        print(f"  Pure Deformation (Bulge/Wobble): {np.mean(micro_procrustes_disps_mm):.3f} mm")
        print(f"  Stretch: {np.mean(stretch_disps_mm):.3f} mm")
        print("\nMAXIMUM DEFORMATION ACROSS ALL FRAMES (mm):")
        print(f"  Max Macro Motion: {np.max(macro_disps_mm):.3f} mm")
        print(f"  Max Pure Deformation: {np.max(micro_procrustes_disps_mm):.3f} mm")
        print(f"  Stretch: {np.max(stretch_disps_mm):.3f} mm")
        print("=" * 42)

    elif ANALYSIS_MODE in ['displacement', 'pure_deformation']:
        for frame_idx, obj_file in enumerate(files_to_process):
            # 1. Load and normalize the current frame's data
            V, F = load_obj(os.path.join(BAKED_MESHES_DIR, obj_file))
            V_norm = (V - offset) / scale  # Apply the SAME normalization

            # 2. Calculate the metric based on the chosen analysis mode
            if ANALYSIS_MODE == 'displacement':
                metric_values_norm = np.linalg.norm(V_norm - V_ref_norm, axis=1)
            elif ANALYSIS_MODE == 'pure_deformation':
                _, V_aligned, _ = trimesh.registration.procrustes(V_norm, V_ref_norm)
                # metric_values_norm = np.linalg.norm(V_aligned - V_ref_norm, axis=1)

                metric_values_norm = compute_signed_distance(V_ref_norm, F_ref, V_aligned)
                metric_values_mm = metric_values_norm * scale * 1000
            else:
                raise ValueError(f"Unsupported ANALYSIS_MODE for video: {ANALYSIS_MODE}")

            # Convert metric to millimeters
            metric_values_mm = metric_values_norm * scale * 1000

            # 3. Define colormap and normalization
            colormap = colormaps['coolwarm']
            # Use LogNorm for a more sensitive color scale. vmin cannot be zero.

            # norm = colors.LogNorm(vmin=LOWER_THRESHOLD, vmax=UPPER_THRESHOLD)
            norm = plt.Normalize(vmin=LOWER_THRESHOLD,
                                 vmax=UPPER_THRESHOLD)

            # Map the metric values to colors
            C = colormap(norm(metric_values_mm))
            C_faces = C[F[:, 0]]

            # 4. Project vertices for rendering
            V_proj = np.c_[V_norm, np.ones(len(V_norm))] @ MVP.T
            V_proj /= V_proj[:, 3].reshape(-1, 1)
            V_proj = V_proj[:, :3]

            # 5. Render the final image with legend
            fig = plt.figure(figsize=(8, 8));
            ax = fig.add_axes([0, 0, 0.85, 1], xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)
            ax.set_xticks([]);
            ax.set_yticks([])
            ax.set_title(f"Sequence: {shot} - Frame {START_FRAME + frame_idx}", color='black', y=0.98)

            visible_faces, visible_colors, face_depths = [], [], []
            V_2d, V_depth = V_proj[:, :2], V_proj[:, 2]
            for i, face in enumerate(F):
                p0, p1, p2 = V_2d[face]
                signed_area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
                if signed_area > 0:
                    visible_faces.append(face);
                    visible_colors.append(C_faces[i]);
                    face_depths.append(np.mean(V_depth[face]))

            if visible_faces:
                sorted_indices = np.argsort(face_depths)[::-1]
                T_sorted = [V_2d[visible_faces[i]] for i in sorted_indices]
                C_sorted = [visible_colors[i] for i in sorted_indices]
                collection = PolyCollection(T_sorted, closed=True, linewidth=0.1, facecolor=C_sorted, edgecolor="black")
                ax.add_collection(collection)

            cax = fig.add_axes([0.7, 0.15, 0.04, 0.7])
            cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=colormap, norm=norm), cax=cax,
                                format=FuncFormatter(format_as_mm))
            cbar.set_label(f'{ANALYSIS_MODE.replace("_", " ").capitalize()} (mm)', color='black', rotation=270,
                           labelpad=20)

            tick_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20]
            cbar.set_ticks(tick_values)

            cax.tick_params(axis='y', colors='black')

            output_filepath = os.path.join(temp_frame_dir, f"{os.path.splitext(obj_file)[0]}.png")
            plt.savefig(output_filepath, dpi=300, facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"Rendered frame for {obj_file}")


            """ OLDER VERSION
            # 2. Calculate the metric based on the chosen analysis mode
            if ANALYSIS_MODE == 'displacement':
                # Fast approximation: measures distance after removing translation.
                metric_values = np.linalg.norm(V_norm - V_ref_norm, axis=1)

                # Convert metric back to original units (mm)
                metric_values_mm = metric_values * scale * 1000

            elif ANALYSIS_MODE == 'pure_deformation':
                # Most accurate: measures distance after removing translation AND rotation.
                _, V_aligned, _ = trimesh.registration.procrustes(V_norm, V_ref_norm)
                metric_values = np.linalg.norm(V_aligned - V_ref_norm, axis=1)

                # Convert metric back to original units (mm)
                metric_values_mm = metric_values * scale * 1000

            elif ANALYSIS_MODE == 'stretch_change':
                # Calculates the average distance to neighboring vertices.
                stretch_current = compute_stretch(V_norm, F)

                # The metric is the DIFFERENCE from the reference stretch
                metric_values = stretch_current - stretch_ref

                # Convert metric back to original units (mm)
                metric_values_mm = metric_values * scale * 1000
            else:
                raise ValueError(f"Unsupported ANALYSIS_MODE for video: {ANALYSIS_MODE}")

            # ---

            if ANALYSIS_MODE == 'stretch_change':
                # Use a diverging colormap for stretch (positive/negative values)
                norm = plt.Normalize(vmin=LOWER_THRESHOLD, vmax=UPPER_THRESHOLD)

                colormap = colormaps['coolwarm']
                C = colormap(norm(metric_values_mm))
                C_faces = C[F[:, 0]]

            elif ANALYSIS_MODE == 'pure_deformation' or ANALYSIS_MODE == 'displacement':
                # 3. Normalize metric for coloring using the defined thresholds
                value_range = UPPER_THRESHOLD - LOWER_THRESHOLD
                if value_range > 0:
                    normalized_colors = (metric_values_mm - LOWER_THRESHOLD) / value_range
                else:
                    normalized_colors = np.zeros_like(metric_values_mm)
                normalized_colors = np.clip(normalized_colors, 0, 1)

                colormap = colormaps['viridis']
                C = colormap(normalized_colors)
                C_faces = C[F[:, 0]]

            # 4. Project vertices for rendering
            V_proj = np.c_[V_norm, np.ones(len(V_norm))] @ MVP.T
            V_proj /= V_proj[:, 3].reshape(-1, 1)
            V_proj = V_proj[:, :3]

            # 5. Render the final image with legend
            fig = plt.figure(figsize=(8, 8));
            ax = fig.add_axes([0, 0, 0.85, 1], xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)
            ax.set_xticks([]);
            ax.set_yticks([])

            ax.set_title(f"Sequence: {shot} - Frame {frame + START_FRAME}")

            visible_faces, visible_colors, face_depths = [], [], []
            V_2d, V_depth = V_proj[:, :2], V_proj[:, 2]
            for i, face in enumerate(F):
                p0, p1, p2 = V_2d[face];
                signed_area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
                if signed_area > 0:
                    visible_faces.append(face);
                    visible_colors.append(C_faces[i]);
                    face_depths.append(np.mean(V_depth[face]))

            if visible_faces:
                sorted_indices = np.argsort(face_depths)[::-1]
                T_sorted = [V_2d[visible_faces[i]] for i in sorted_indices]
                C_sorted = [visible_colors[i] for i in sorted_indices]
                collection = PolyCollection(T_sorted, closed=True, linewidth=0.1, facecolor=C_sorted, edgecolor="black")
                ax.add_collection(collection)

            cax = fig.add_axes([0.7, 0.15, 0.04, 0.7])
            # norm = plt.Normalize(vmin=LOWER_THRESHOLD, vmax=UPPER_THRESHOLD)

            cbar = fig.colorbar(
                plt.cm.ScalarMappable(cmap=colormap, norm=norm),
                cax=cax,
                orientation='vertical',
                format=FuncFormatter(format_as_mm)  # Use the new meter formatter
            )

            if ANALYSIS_MODE == 'pure_deformation' or ANALYSIS_MODE == 'displacement':
                cbar.set_label('Pure Deformation (mm)', color='black', rotation=270, labelpad=20)
            elif ANALYSIS_MODE == 'stretch_change':
                cbar.set_label('Local Stretch (mm)', color='black', rotation=270, labelpad=20)

            cax.tick_params(axis='y', colors='black')

            output_filepath = os.path.join(temp_frame_dir, f"{os.path.splitext(obj_file)[0]}.png")
            plt.savefig(output_filepath, dpi=300, facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"Rendered frame for {obj_file}")
            """

        create_video(temp_frame_dir, OUTPUT_PATH, VIDEO_FPS, CLEANUP_PNGS)

    elif ANALYSIS_MODE == 'stretch_change':

        for frame_idx, obj_file in enumerate(files_to_process):
            # 1. Load and normalize the current frame's data
            V, F = load_obj(os.path.join(BAKED_MESHES_DIR, obj_file))
            V_norm = (V - offset) / scale  # Apply the SAME normalization

            # Calculates the average distance to neighboring vertices.
            stretch_current = compute_stretch(V_norm, F)

            # The metric is the DIFFERENCE from the reference stretch
            metric_values = stretch_current - stretch_ref

            # Convert metric back to original units (mm)
            metric_values_mm = metric_values * scale * 1000

            # Use a diverging colormap for stretch (positive/negative values)
            norm = plt.Normalize(vmin=LOWER_THRESHOLD, vmax=UPPER_THRESHOLD)

            colormap = colormaps['coolwarm']
            C = colormap(norm(metric_values_mm))
            C_faces = C[F[:, 0]]

            # 4. Project vertices for rendering
            V_proj = np.c_[V_norm, np.ones(len(V_norm))] @ MVP.T
            V_proj /= V_proj[:, 3].reshape(-1, 1)
            V_proj = V_proj[:, :3]

            # 5. Render the final image with legend
            fig = plt.figure(figsize=(8, 8));
            ax = fig.add_axes([0, 0, 0.85, 1], xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)
            ax.set_xticks([]);
            ax.set_yticks([])

            ax.set_title(f"Sequence: {shot} - Frame {frame_idx + START_FRAME}")

            visible_faces, visible_colors, face_depths = [], [], []
            V_2d, V_depth = V_proj[:, :2], V_proj[:, 2]
            for i, face in enumerate(F):
                p0, p1, p2 = V_2d[face];
                signed_area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
                if signed_area > 0:
                    visible_faces.append(face);
                    visible_colors.append(C_faces[i]);
                    face_depths.append(np.mean(V_depth[face]))

            if visible_faces:
                sorted_indices = np.argsort(face_depths)[::-1]
                T_sorted = [V_2d[visible_faces[i]] for i in sorted_indices]
                C_sorted = [visible_colors[i] for i in sorted_indices]
                collection = PolyCollection(T_sorted, closed=True, linewidth=0.1, facecolor=C_sorted, edgecolor="black")
                ax.add_collection(collection)

            cax = fig.add_axes([0.7, 0.15, 0.04, 0.7])
            # norm = plt.Normalize(vmin=LOWER_THRESHOLD, vmax=UPPER_THRESHOLD)

            cbar = fig.colorbar(
                plt.cm.ScalarMappable(cmap=colormap, norm=norm),
                cax=cax,
                orientation='vertical',
                format=FuncFormatter(format_as_mm)  # Use the new meter formatter
            )

            cbar.set_label('Local Stretch (mm)', color='black', rotation=270, labelpad=20)

            cax.tick_params(axis='y', colors='black')

            output_filepath = os.path.join(temp_frame_dir, f"{os.path.splitext(obj_file)[0]}.png")
            plt.savefig(output_filepath, dpi=300, facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"Rendered frame for {obj_file}")

        # --- 5. Compile Video ---
        create_video(temp_frame_dir, OUTPUT_PATH, VIDEO_FPS, CLEANUP_PNGS)

    else:
        raise ValueError(f"Unsupported ANALYSIS_MODE: {ANALYSIS_MODE}")


# --- 4. SCRIPT EXECUTION ---

if __name__ == "__main__":
    # This loop will run the entire analysis for each shot in the SHOT_LIST.
    for shot_to_process in SHOT_LIST:
        create_shot(shot_to_process)

    # BASE_DIR = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans"
    # OUTPUT_DIRECTORY = os.path.join(BASE_DIR, "analysis", 'shot_002')
    # temp_frame_dir = os.path.join(OUTPUT_DIRECTORY, "temp_frames")
    # OUTPUT_PATH = os.path.join(OUTPUT_DIRECTORY, f"shot_002_pure_deformation_analysis.mp4")
    #
    # create_video(temp_frame_dir, OUTPUT_PATH, 24, False)

    print("\nAll shots processed.")