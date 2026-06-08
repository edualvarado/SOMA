import time
import viser
import numpy as np
import json
import os
import trimesh
from scipy.spatial.transform import Rotation

def _rot_x(points, deg):
    if deg == 0.0: return points
    R = Rotation.from_euler('x', deg, degrees=True).as_matrix()
    return points @ R.T

def main():
    # --- CONFIGURATION ---
    # Change this to S1 to debug your regression, or S2 to fix the starburst
    SUBJECT = "S3" 
    
    BASE_DIR = rf"/CT/SOMA/static00/{SUBJECT}"
    MESH_PATH = os.path.join(BASE_DIR, "layers", "tpose", f"skin_layer-{SUBJECT}-TPose.obj")
    
    # Handle S1 vs S2 naming differences
    if SUBJECT == "S1":
        MARKER_PATH = os.path.join(BASE_DIR, "canonical_model", "canonical_data_tpose.json")
        INITIAL_SCALE = 1.0
    else:
        MARKER_PATH = os.path.join(BASE_DIR, "canonical_model", f"{SUBJECT}_canonical_data_tpose.json")
        INITIAL_SCALE = 0.001 # Start with the fix for S2

    server = viser.ViserServer()
    server.scene.add_grid("grid", plane="xz")
    server.scene.set_up_direction("+y")

    # --- Load Data ---
    print(f"Loading {SUBJECT}...")
    mesh = trimesh.load(MESH_PATH, process=False)
    
    with open(MARKER_PATH, 'r') as f:
        data = json.load(f)
        canonical_data = data.get("0", data)
    
    # Flatten markers
    raw_markers = []
    for mid in sorted(canonical_data.keys()):
        raw_markers.append(canonical_data[mid][0])
    raw_markers = np.array(raw_markers)

    # --- GUI ---
    with server.gui.add_folder("Alignment Controls"):
        scale_slider = server.gui.add_slider("Marker Scale", min=0.0001, max=1.5, step=0.0001, initial_value=INITIAL_SCALE)
        
        marker_rot_x = server.gui.add_slider("Marker Rot X", min=-180, max=180, step=90, initial_value=0)
        marker_rot_y = server.gui.add_slider("Marker Rot Y", min=-180, max=180, step=90, initial_value=0)
        
        mesh_rot_x = server.gui.add_slider("Mesh Rot X", min=-180, max=180, step=90, initial_value=0)
        
        info_text = server.gui.add_text("Info", initial_value="Adjust sliders until dots match skin.")

    # --- Update Loop ---
    def update_scene():
        # 1. Mesh Transformation
        # We don't modify the trimesh object in a loop (expensive), we just rotate the visualization
        m_rot = mesh_rot_x.value
        if m_rot != 0:
            R = Rotation.from_euler('x', m_rot, degrees=True).as_matrix()
            # Apply to vertices just for viz
            viz_verts = mesh.vertices @ R.T
        else:
            viz_verts = mesh.vertices

        server.scene.add_mesh_simple(
            "/skin",
            vertices=viz_verts,
            faces=mesh.faces,
            color=(200, 200, 200),
            opacity=0.8
        )

        # 2. Marker Transformation
        s = scale_slider.value
        rx = marker_rot_x.value
        ry = marker_rot_y.value
        
        scaled_markers = raw_markers * s
        
        # Apply Rotations
        if rx != 0:
            scaled_markers = _rot_x(scaled_markers, rx)
        if ry != 0:
            # Simple Y rot
            R_y = Rotation.from_euler('y', ry, degrees=True).as_matrix()
            scaled_markers = scaled_markers @ R_y.T
            
        server.scene.add_point_cloud(
            "/markers",
            points=scaled_markers,
            colors=(255, 0, 0),
            point_size=0.005
        )
        
        info_text.value = f"Scale: {s} | MarkerRotX: {rx} | MeshRotX: {m_rot}"

    # Listeners
    scale_slider.on_update(lambda _: update_scene())
    marker_rot_x.on_update(lambda _: update_scene())
    marker_rot_y.on_update(lambda _: update_scene())
    mesh_rot_x.on_update(lambda _: update_scene())

    update_scene()
    print("Visualizer running at http://localhost:8080")
    
    while True:
        time.sleep(0.1)

if __name__ == "__main__":
    main()