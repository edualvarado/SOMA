import bpy
import numpy as np

# --- CONFIGURATION ---
# Type the EXACT names of the objects as they appear in your Blender Outliner
TARGET_OBJ_NAME = "m_final_tpose_75"  # The mesh that will get the heatmap
REF_OBJ_NAME = "m_final_tpose_0"       # The rest pose / reference mesh

LIMIT = 0.007  # Sensitivity (meters). 0.005 = 5mm range for full color
LAYER_NAME = "Deformation_Map"
# ---------------------

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
        
        node_attr = nodes.new('ShaderNodeAttribute') 
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
    
    # 1. Grab Objects from the scene
    obj_target = bpy.data.objects.get(TARGET_OBJ_NAME)
    obj_source = bpy.data.objects.get(REF_OBJ_NAME)
    
    if not obj_target:
        print(f"[ERROR] Target object '{TARGET_OBJ_NAME}' not found in scene!")
        return
    if not obj_source:
        print(f"[ERROR] Reference object '{REF_OBJ_NAME}' not found in scene!")
        return
        
    # Ensure they are meshes
    if obj_target.type != 'MESH' or obj_source.type != 'MESH':
        print("[ERROR] Both objects must be Meshes!")
        return
    
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
    colors = np.ones((n_verts, 4), dtype=np.float32) # Default White (R,G,B,A)
    
    # Positive (Red Outward Bulge)
    mask_pos = factors > 0
    f_pos = factors[mask_pos]
    colors[mask_pos, 1] = 1.0 - f_pos # Reduce Green
    colors[mask_pos, 2] = 1.0 - f_pos # Reduce Blue
    
    # Negative (Blue Inward Compression)
    mask_neg = factors < 0
    f_neg = np.abs(factors[mask_neg])
    colors[mask_neg, 0] = 1.0 - f_neg # Reduce Red
    colors[mask_neg, 1] = 1.0 - f_neg # Reduce Green

    # 5. Apply to Target using modern API
    mesh = obj_target.data
    
    if hasattr(mesh, "color_attributes"):
        # Blender 3.2+
        if LAYER_NAME in mesh.color_attributes:
            mesh.color_attributes.remove(mesh.color_attributes[LAYER_NAME])
            
        # POINT domain maps 1:1 with vertices
        color_attr = mesh.color_attributes.new(name=LAYER_NAME, type='FLOAT_COLOR', domain='POINT')
        color_attr.data.foreach_set("color", colors.flatten())
        mesh.color_attributes.active_color = color_attr
    else:
        # Fallback for old Blender versions
        if LAYER_NAME in mesh.vertex_colors:
            mesh.vertex_colors.remove(mesh.vertex_colors[LAYER_NAME])
        color_attr = mesh.vertex_colors.new(name=LAYER_NAME)
        
        loop_count = len(mesh.loops)
        loop_v_indices = np.zeros(loop_count, dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", loop_v_indices)
        
        final_colors = colors[loop_v_indices].flatten()
        color_attr.data.foreach_set("color", final_colors)
    
    # 6. Setup Material automatically
    create_heatmap_material(obj_target, LAYER_NAME)
    
    # Optional: Hide the source so you can clearly see the target
    obj_source.hide_viewport = True
    
    print(f"[SUCCESS] Heatmap applied to '{obj_target.name}'.")
    print("Switch Viewport to 'Material Preview' to see it.")

if __name__ == "__main__":
    main()