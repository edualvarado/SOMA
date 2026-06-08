import bpy
import bmesh
import numpy as np

# --- CONFIGURATION ---
RIGGED_OBJ_NAME = "Final_Render_Mesh_40_340_shot_004"
SEQUENCE_CONTAINER = "exported_musc_meshes_40_340_shot_004" # The object holding the sequence data
HEATMAP_LAYER_NAME = "Deformation_Map"     # Name of the color attribute
LIMIT = 0.0025                             # Sensitivity (in meters). 0.005 = 5mm
# ---------------------

def update_animation_and_heatmap(scene):
    # 1. Get Objects
    rigged_obj = bpy.data.objects.get(RIGGED_OBJ_NAME)
    seq_obj = bpy.data.objects.get(SEQUENCE_CONTAINER)
    
    if not rigged_obj or not seq_obj: return
    if rigged_obj.type != 'MESH' or seq_obj.type != 'MESH': return

    mesh_rigged = rigged_obj.data
    mesh_seq = seq_obj.data
    
    n_verts = len(mesh_rigged.vertices)
    if len(mesh_seq.vertices) != n_verts: return

    # --- PART A: UPDATE GEOMETRY ---
    current_coords = np.zeros(n_verts * 3, dtype=np.float32)
    mesh_seq.vertices.foreach_get("co", current_coords)
    mesh_rigged.vertices.foreach_set("co", current_coords)
    
    # --- PART B: UPDATE HEATMAP ---
    if "rest_coords" not in rigged_obj:
        rigged_obj["rest_coords"] = current_coords
        print("[INIT] Reference Pose Captured")
    
    rest_coords = rigged_obj["rest_coords"]
    
    # Calculate Displacement
    P_curr = current_coords.reshape((-1, 3))
    P_rest = np.array(rest_coords).reshape((-1, 3))
    diff = P_curr - P_rest
    dists = np.linalg.norm(diff, axis=1)
    
    # Calculate Sign (Bulge vs Shrink)
    normals = np.zeros(n_verts * 3, dtype=np.float32)
    mesh_seq.vertices.foreach_get("normal", normals)
    normals = normals.reshape((-1, 3))
    signs = np.einsum('ij,ij->i', diff, normals)
    signed_dists = dists * np.sign(signs)
    
    # --- COLOR LOGIC (WHITE BASE) ---
    factors = np.clip(signed_dists / LIMIT, -1.0, 1.0)
    
    # 1. Initialize everything to WHITE (1.0, 1.0, 1.0)
    # This ensures stable areas (head/hands) are White, not Yellow/Green.
    colors = np.ones((n_verts, 4), dtype=np.float32) 
    
    # 2. Positive (Bulge) -> White to Red
    # We keep Red at 1.0, and decrease Green & Blue
    mask_pos = factors > 0
    f_pos = factors[mask_pos]
    colors[mask_pos, 1] = 1.0 - f_pos  # Green fades out
    colors[mask_pos, 2] = 1.0 - f_pos  # Blue fades out
    
    # 3. Negative (Shrink) -> White to Blue
    # We keep Blue at 1.0, and decrease Red & Green
    mask_neg = factors < 0
    f_neg = np.abs(factors[mask_neg])
    colors[mask_neg, 0] = 1.0 - f_neg  # Red fades out
    colors[mask_neg, 1] = 1.0 - f_neg  # Green fades out

    # Write Attribute
    if not mesh_rigged.vertex_colors.get(HEATMAP_LAYER_NAME):
        mesh_rigged.vertex_colors.new(name=HEATMAP_LAYER_NAME)
    
    color_layer = mesh_rigged.vertex_colors[HEATMAP_LAYER_NAME]
    
    loop_count = len(mesh_rigged.loops)
    loop_v_indices = np.zeros(loop_count, dtype=np.int32)
    mesh_rigged.loops.foreach_get("vertex_index", loop_v_indices)
    loop_colors = colors[loop_v_indices].flatten()
    
    color_layer.data.foreach_set("color", loop_colors)
    mesh_rigged.update()

# --- REGISTER ---
bpy.app.handlers.frame_change_post.clear()
bpy.app.handlers.frame_change_post.append(update_animation_and_heatmap)