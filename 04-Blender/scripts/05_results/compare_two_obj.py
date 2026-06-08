import bpy
import os
import numpy as np

# --- CONFIGURATION ---
# Path to the "Rest Pose" or original file
PATH_SOURCE = r"T:\static00\S1\exported_ind_musc_meshes_0_300_shot_001\S1\shot_001_captury\frame_0000\l-vastus-lateralis_t-pose.obj"

# Path to the "Deformed" file you want to analyze
PATH_TARGET = r"T:\static00\S1\exported_ind_musc_meshes_0_300_shot_001\S1\shot_001_captury\frame_0160\l-vastus-lateralis_t-pose.obj"

LIMIT = 0.005  # Sensitivity (meters). 0.005 = 5mm range for full color
LAYER_NAME = "Deformation_Map"
# ---------------------

def load_obj(filepath):
    """Robust OBJ loader for Blender 3.6 and 4.0+"""
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return None

    # Deselect all
    bpy.ops.object.select_all(action='DESELECT')

    try:
        if bpy.app.version >= (4, 0, 0):
            bpy.ops.wm.obj_import(filepath=filepath)
        else:
            bpy.ops.import_scene.obj(filepath=filepath, use_split_objects=False)
            
        if bpy.context.selected_objects:
            obj = bpy.context.selected_objects[0]
            return obj
    except Exception as e:
        print(f"[ERROR] Import failed: {e}")
        return None

def create_heatmap_material(obj, attribute_name):
    """Auto-generates a material that displays the Vertex Color attribute"""
    mat_name = "Heatmap_Material"
    
    # Get or create material
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        
        # Nodes
        node_out = nodes.new('ShaderNodeOutputMaterial')
        node_out.location = (300, 0)
        
        node_bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        node_bsdf.location = (0, 0)
        
        node_attr = nodes.new('ShaderNodeAttribute') # Generic Attribute Node
        node_attr.location = (-300, 0)
        node_attr.attribute_type = 'GEOMETRY'
        node_attr.attribute_name = attribute_name
        
        # Links
        links.new(node_attr.outputs['Color'], node_bsdf.inputs['Base Color'])
        links.new(node_bsdf.outputs['BSDF'], node_out.inputs['Surface'])
    
    # Assign to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

def main():
    print("--- STARTING COMPARISON ---")
    
    # 1. Load Objects
    print("Loading Source...")
    obj_source = load_obj(PATH_SOURCE)
    if not obj_source: return
    obj_source.name = "Source_Ref"
    
    print("Loading Target...")
    obj_target = load_obj(PATH_TARGET)
    if not obj_target: return
    obj_target.name = "Target_Deformed"
    
    # Move source slightly so they don't z-fight (Optional)
    obj_source.location.x -= 1.0 
    
    # 2. Get Data
    n_verts = len(obj_target.data.vertices)
    if len(obj_source.data.vertices) != n_verts:
        print(f"[ERROR] Topology mismatch! Source: {len(obj_source.data.vertices)}, Target: {n_verts}")
        return

    coords_src = np.zeros(n_verts * 3, dtype=np.float32)
    coords_tgt = np.zeros(n_verts * 3, dtype=np.float32)
    normals_tgt = np.zeros(n_verts * 3, dtype=np.float32)
    
    obj_source.data.vertices.foreach_get("co", coords_src)
    obj_target.data.vertices.foreach_get("co", coords_tgt)
    obj_target.data.vertices.foreach_get("normal", normals_tgt)
    
    # 3. Calculate Math
    P_src = coords_src.reshape((-1, 3))
    P_tgt = coords_tgt.reshape((-1, 3))
    N_tgt = normals_tgt.reshape((-1, 3))
    
    diff = P_tgt - P_src
    dists = np.linalg.norm(diff, axis=1)
    
    # Sign (Bulge vs Shrink)
    signs = np.einsum('ij,ij->i', diff, N_tgt)
    signed_dists = dists * np.sign(signs)
    
    # 4. Color Logic (White Base)
    factors = np.clip(signed_dists / LIMIT, -1.0, 1.0)
    colors = np.ones((n_verts, 4), dtype=np.float32) # Default White
    
    # Positive (Red)
    mask_pos = factors > 0
    f_pos = factors[mask_pos]
    colors[mask_pos, 1] = 1.0 - f_pos # Reduce Green
    colors[mask_pos, 2] = 1.0 - f_pos # Reduce Blue
    
    # Negative (Blue)
    mask_neg = factors < 0
    f_neg = np.abs(factors[mask_neg])
    colors[mask_neg, 0] = 1.0 - f_neg # Reduce Red
    colors[mask_neg, 1] = 1.0 - f_neg # Reduce Green

    # 5. Apply to Target
    mesh = obj_target.data
    if not mesh.vertex_colors.get(LAYER_NAME):
        mesh.vertex_colors.new(name=LAYER_NAME)
    
    color_layer = mesh.vertex_colors[LAYER_NAME]
    
    loop_count = len(mesh.loops)
    loop_v_indices = np.zeros(loop_count, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", loop_v_indices)
    
    final_colors = colors[loop_v_indices].flatten()
    color_layer.data.foreach_set("color", final_colors)
    
    # 6. Setup Material automatically
    create_heatmap_material(obj_target, LAYER_NAME)
    
    print(f"[SUCCESS] Heatmap applied to '{obj_target.name}'.")
    print("Switch Viewport to 'Material Preview' to see it.")

if __name__ == "__main__":
    main()