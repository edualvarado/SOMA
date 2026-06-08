import bpy
import numpy as np

# --- CONFIGURATION ---
TARGET_OBJ_NAME = bpy.context.active_object.name
LIMIT = 0.006              # Sensitivity (meters)


# Type the exact name of the muscle you want to isolate for the heatmap
# All other muscles will show their ID colors + Wireframe
# ISOLATE_MUSCLE_NAME = "l-pectoralis-major_t-pose"

ISOLATE_MUSCLE_NAME = "ALL"

# ISOLATE_MUSCLE_NAME = "None" # ONLY FOR INDIVIDUAL MUSCLES NO HEATMAP

SHOW_WIREFRAME = True

# ---------------------

LAYER_NAME = "Deformation_Map"   # The heatmap layer
BASE_LAYER_NAME = "ID_Color"     # The layer with the random muscle colors

if "soma_heatmap_cache" not in locals():
    soma_heatmap_cache = {}

def create_heatmap_material(obj):
    """Generates a safe shader using a Mix Node to overlay the wireframe"""
    mat_name = "Hybrid_Heatmap_Wireframe_Material"
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        
        node_out = nodes.new('ShaderNodeOutputMaterial')
        node_out.location = (600, 0)
        node_bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        node_bsdf.location = (300, 0)
        
        # 1. Base Colors
        node_color = nodes.new('ShaderNodeAttribute') 
        node_color.location = (-300, 0)
        node_color.attribute_type = 'GEOMETRY'
        node_color.attribute_name = LAYER_NAME
        
        # 2. Wireframe Mask
        node_mask = nodes.new('ShaderNodeAttribute')
        node_mask.location = (-300, -200)
        node_mask.attribute_type = 'GEOMETRY'
        node_mask.attribute_name = "Wireframe_Mask"
        
        # 3. GPU Wireframe Node
        node_wire = nodes.new('ShaderNodeWireframe')
        node_wire.location = (-300, -400)
        node_wire.inputs[0].default_value = 0.001 # Wire thickness
        
        # Multiply Native Wireframe by our Custom Mask
        node_math_mult = nodes.new('ShaderNodeMath')
        node_math_mult.operation = 'MULTIPLY'
        node_math_mult.location = (0, -250)
        links.new(node_wire.outputs[0], node_math_mult.inputs[0])
        links.new(node_mask.outputs['Fac'], node_math_mult.inputs[1])
        
        # 4. Mix Node
        if bpy.app.version >= (3, 4, 0):
            node_mix = nodes.new('ShaderNodeMix')
            node_mix.data_type = 'RGBA'
            fac_sock = node_mix.inputs[0]
            col1_sock = node_mix.inputs[6] 
            col2_sock = node_mix.inputs[7]
        else:
            node_mix = nodes.new('ShaderNodeMixRGB')
            fac_sock = node_mix.inputs[0]
            col1_sock = node_mix.inputs[1]
            col2_sock = node_mix.inputs[2]
            
        node_mix.location = (100, 0)
        col2_sock.default_value = (0.02, 0.02, 0.02, 1.0) # Black wireframe color
        
        # Link Mix Node: Factor=WireframeMask, A=VertexColors, B=Black
        links.new(node_math_mult.outputs[0], fac_sock)
        links.new(node_color.outputs['Color'], col1_sock)
        
        # Plug into BSDF
        links.new(node_mix.outputs[0], node_bsdf.inputs['Base Color'])
        links.new(node_bsdf.outputs['BSDF'], node_out.inputs['Surface'])
    
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

def cache_reference_state():
    """Captures the Rest pose and sets up the static shader attributes."""
    global soma_heatmap_cache
    obj = bpy.data.objects.get(TARGET_OBJ_NAME)
    if not obj or obj.type != 'MESH':
        print(f"[ERROR] Target object '{TARGET_OBJ_NAME}' not found.")
        return False
        
    mesh = obj.data
    n_verts = len(mesh.vertices)
    
    ref_coords = np.zeros(n_verts * 3, dtype=np.float32)
    ref_normals = np.zeros(n_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", ref_coords)
    mesh.vertices.foreach_get("normal", ref_normals)
    
    soma_heatmap_cache['coords'] = ref_coords.reshape((-1, 3))
    soma_heatmap_cache['normals'] = ref_normals.reshape((-1, 3))
    soma_heatmap_cache['n_verts'] = n_verts
    
    # Extract Base ID Colors
    base_colors = np.ones((n_verts, 4), dtype=np.float32)
    if hasattr(mesh, "color_attributes") and BASE_LAYER_NAME in mesh.color_attributes:
        flat_colors = np.zeros(n_verts * 4, dtype=np.float32)
        mesh.color_attributes[BASE_LAYER_NAME].data.foreach_get("color", flat_colors)
        base_colors = flat_colors.reshape((n_verts, 4))
    else:
        print(f"[WARNING] Could not find '{BASE_LAYER_NAME}'. Run the Colorizer script first!")
    
    soma_heatmap_cache['base_colors'] = base_colors

    # # Figure out mask for the isolated muscle
    # active_mask = np.ones(n_verts, dtype=bool) 
    # if ISOLATE_MUSCLE_NAME:
    #     if ISOLATE_MUSCLE_NAME in obj.vertex_groups:
    #         active_mask = np.zeros(n_verts, dtype=bool)
    #         vg_idx = obj.vertex_groups[ISOLATE_MUSCLE_NAME].index
    #         for v in mesh.vertices:
    #             for g in v.groups:
    #                 if g.group == vg_idx:
    #                     active_mask[v.index] = True
    #                     break
    #     else:
    #         print(f"[WARNING] Muscle '{ISOLATE_MUSCLE_NAME}' not found in Vertex Groups!")

    # --- UPDATED: Mask Logic ---
    # Default to all False (NO heatmap applied anywhere)
    active_mask = np.zeros(n_verts, dtype=bool) 
    
    # Only try to apply heatmap if the string is valid and NOT "None"
    if ISOLATE_MUSCLE_NAME and ISOLATE_MUSCLE_NAME.lower() != "none":
        
        # --- NEW: Check for "ALL" keyword ---
        if ISOLATE_MUSCLE_NAME.upper() == "ALL":
            active_mask = np.ones(n_verts, dtype=bool) # Select every vertex
            print("[INFO] Applying heatmap to ALL muscles.")
        # ------------------------------------
            
        elif ISOLATE_MUSCLE_NAME in obj.vertex_groups:
            vg_idx = obj.vertex_groups[ISOLATE_MUSCLE_NAME].index
            for v in mesh.vertices:
                for g in v.groups:
                    if g.group == vg_idx:
                        active_mask[v.index] = True
                        break
            print(f"[INFO] Isolating muscle '{ISOLATE_MUSCLE_NAME}' for heatmap.")
        else:
            print(f"[WARNING] Muscle '{ISOLATE_MUSCLE_NAME}' not found! Showing ID colors only.")
    else:
        print("[INFO] No muscle isolated. Showing full ID colors and wireframe.")
    # ---------------------------
            
    soma_heatmap_cache['active_mask'] = active_mask
    
    # Setup Heatmap Layer
    if hasattr(mesh, "color_attributes"):
        if LAYER_NAME not in mesh.color_attributes:
            mesh.color_attributes.new(name=LAYER_NAME, type='FLOAT_COLOR', domain='POINT')
            
    # Write the static Wireframe Mask to the mesh
    if "Wireframe_Mask" not in mesh.attributes:
        mesh.attributes.new(name="Wireframe_Mask", type='FLOAT', domain='POINT')
        
    # Default to 1.0 (Wireframe ON for everything)
    mask_array = np.ones(n_verts, dtype=np.float32)
    
    # Turn OFF wireframe (0.0) ONLY for the isolated muscle,
    # but keep it ON if we are showing "ALL" so we can see the muscle boundaries!
    if ISOLATE_MUSCLE_NAME and ISOLATE_MUSCLE_NAME.lower() != "none" and ISOLATE_MUSCLE_NAME.upper() != "ALL":
        mask_array[active_mask] = 0.0

    # --- NEW: Respect Global Wireframe Toggle ---
    if not SHOW_WIREFRAME:
        mask_array = np.zeros(n_verts, dtype=np.float32)
    # --------------------------------------------
        
    mesh.attributes["Wireframe_Mask"].data.foreach_set("value", mask_array)
    
    create_heatmap_material(obj)
    print(f"[SUCCESS] Cached Frame {bpy.context.scene.frame_current} as Reference.")
    return True

def dynamic_heatmap_handler(scene):
    """Runs extremely fast vector math to paint the mesh."""
    global soma_heatmap_cache
    if not soma_heatmap_cache:
        return
        
    obj = bpy.data.objects.get(TARGET_OBJ_NAME)
    if not obj: return
    
    mesh = obj.data
    n_verts = soma_heatmap_cache['n_verts']
    
    curr_coords = np.zeros(n_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", curr_coords)
    curr_coords = curr_coords.reshape((-1, 3))
    
    diff = curr_coords - soma_heatmap_cache['coords']
    dists = np.linalg.norm(diff, axis=1)
    
    signs = np.sign(np.einsum('ij,ij->i', diff, soma_heatmap_cache['normals']))
    signed_dists = dists * signs
    
    factors = np.clip(signed_dists / LIMIT, -1.0, 1.0)
    
    # Start with the beautiful background ID colors
    colors = soma_heatmap_cache['base_colors'].copy()
    active_verts = soma_heatmap_cache['active_mask']
    
    # --- CRITICAL FIX ---
    # Wipe the random ID color from the isolated muscle and make it pure White
    # so the Red/Blue heatmap math has a clean canvas!
    colors[active_verts] = [1.0, 1.0, 1.0, 1.0]
    # --------------------
    
    # Positive (Red Outward) - Apply ONLY to isolated muscle
    mask_pos = (factors > 0) & active_verts
    f_pos = factors[mask_pos]
    colors[mask_pos, 0] = 1.0         
    colors[mask_pos, 1] = 1.0 - f_pos 
    colors[mask_pos, 2] = 1.0 - f_pos 
    
    # Negative (Blue Inward) - Apply ONLY to isolated muscle
    mask_neg = (factors < 0) & active_verts
    f_neg = np.abs(factors[mask_neg])
    colors[mask_neg, 0] = 1.0 - f_neg 
    colors[mask_neg, 1] = 1.0 - f_neg 
    colors[mask_neg, 2] = 1.0         

    color_attr = mesh.color_attributes[LAYER_NAME]
    color_attr.data.foreach_set("color", colors.flatten())
    mesh.update()

# --- EXECUTION ---
bpy.app.handlers.frame_change_post.clear()

if cache_reference_state():
    bpy.app.handlers.frame_change_post.append(dynamic_heatmap_handler)
    print("--- High-Speed Heatmap & Wireframe Context Active! Press Play. ---")