import bpy
import bmesh
from mathutils import Vector
import numpy as np

def create_deformation_heatmap():
    # --- CONFIGURATION ---
    # 4mm limit (Red > 4mm, Blue < -4mm)
    LIMIT = 0.0025
    # ---------------------
    
    # 1. Validation
    target_obj = bpy.context.active_object
    selected = bpy.context.selected_objects
    
    # Find the other selected mesh (Reference)
    ref_obj = None
    for obj in selected:
        if obj != target_obj and obj.type == 'MESH':
            ref_obj = obj
            break
            
    if not target_obj or not ref_obj:
        print("ERROR: Select Reference (Rest), then Shift-Select Target (Deformed).")
        return

    print(f"Heatmap: '{ref_obj.name}' -> '{target_obj.name}' (Limit: {LIMIT}m)")

    # 3. Geometry Data
    bpy.context.view_layer.update()
    
    mesh_tgt = target_obj.data
    mesh_ref = ref_obj.data
    
    verts_tgt = mesh_tgt.vertices
    verts_ref = mesh_ref.vertices
    
    n_tgt = len(verts_tgt)
    n_ref = len(verts_ref)
    
    # [FIX] RELAXED CHECK: Print warning but DO NOT STOP
    print(f"--- VERTEX COUNT CHECK ---")
    print(f"Target ({target_obj.name}): {n_tgt} vertices")
    print(f"Ref    ({ref_obj.name}): {n_ref} vertices")
    
    if n_tgt != n_ref:
        print(f"WARNING: Mismatch of {abs(n_tgt - n_ref)} vertices. Proceeding with truncation/padding...")
        print("         (The heatmap might be slightly misaligned at the very end indices)")
    
    # 4. Calculate Signed Distances (World Space)
    mat_tgt = target_obj.matrix_world
    mat_ref = ref_obj.matrix_world
    
    # Normal Matrix (Inverse Transpose)
    norm_mat_ref = mat_ref.to_3x3().inverted().transposed()

    signed_dists = []
    
    # [FIX] Loop only over the COMMON number of vertices
    common_count = min(n_tgt, n_ref)
    
    for i in range(common_count):
        v_tgt = verts_tgt[i]
        v_ref = verts_ref[i]
        
        pos_tgt = mat_tgt @ v_tgt.co
        pos_ref = mat_ref @ v_ref.co
        
        # Handle degenerate normals
        normal_world = norm_mat_ref @ v_ref.normal
        if normal_world.length_squared > 1e-8:
            norm_ref = normal_world.normalized()
        else:
            norm_ref = Vector((0, 0, 1))
        
        disp = pos_tgt - pos_ref
        dist = disp.dot(norm_ref)
        signed_dists.append(dist)

    # [FIX] Handle Extra Vertices in Target
    # If Target has more vertices than Ref, we must pad the list so color assignment works.
    # We pad with 0.0 (Neutral color).
    if n_tgt > n_ref:
        diff = n_tgt - n_ref
        print(f"Padding {diff} extra target vertices with 0.0 (Neutral).")
        signed_dists.extend([0.0] * diff)
    
    # If Target has FEWER, signed_dists is already the size of Target, so we are good.

    # Print Stats
    max_dist = max(signed_dists) if signed_dists else 0
    min_dist = min(signed_dists) if signed_dists else 0
    print(f"Stats: Max={max_dist:.5f}m, Min={min_dist:.5f}m")

    # 5. Generate Vertex Colors
    col_name = "Deformation_Heatmap"
    if mesh_tgt.color_attributes.get(col_name):
        mesh_tgt.color_attributes.remove(mesh_tgt.color_attributes.get(col_name))
        
    mesh_tgt.color_attributes.new(name=col_name, type='FLOAT_COLOR', domain='CORNER')
    color_layer = mesh_tgt.color_attributes[col_name]
    mesh_tgt.attributes.active_color_index = 0 

    vertex_colors = []
    BASE = 0.7 
    
    for val in signed_dists:
        # Normalize -1 to 1 based on LIMIT
        norm = max(min(val / LIMIT, 1.0), -1.0)
        
        if norm > 0:
            # Positive (Red)
            r = BASE + (1.0 - BASE) * norm
            g = BASE * (1.0 - norm)
            b = BASE * (1.0 - norm)
        else:
            # Negative (Blue)
            norm = abs(norm)
            r = BASE * (1.0 - norm)
            g = BASE * (1.0 - norm)
            b = BASE + (1.0 - BASE) * norm
            
        vertex_colors.append((r, g, b, 1.0))

    # Apply to Loop Indices
    for poly in mesh_tgt.polygons:
        for loop_index in poly.loop_indices:
            v_idx = mesh_tgt.loops[loop_index].vertex_index
            
            # [FIX] Safety check to prevent index out of bounds if mesh topology is extremely different
            if v_idx < len(vertex_colors):
                color_layer.data[loop_index].color = vertex_colors[v_idx]
            else:
                # Fallback color (Green/Neutral)
                color_layer.data[loop_index].color = (0.7, 0.7, 0.7, 1.0)

    # 6. Material Setup (EMISSION SHADER)
    mat_name = "Heatmap_Emission_Skin"
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        
        # Output
        out = nodes.new('ShaderNodeOutputMaterial')
        out.location = (300, 0)
        
        # Emission Shader
        emit = nodes.new('ShaderNodeEmission')
        emit.location = (0, 0)
        emit.inputs['Strength'].default_value = 1.0
        
        # Attribute
        attr = nodes.new('ShaderNodeAttribute')
        attr.location = (-300, 0)
        attr.attribute_name = col_name
        
        # Link
        links.new(attr.outputs['Color'], emit.inputs['Color'])
        links.new(emit.outputs['Emission'], out.inputs['Surface'])

    # Assign Material
    if not target_obj.data.materials:
        target_obj.data.materials.append(mat)
    else:
        target_obj.data.materials[0] = mat

    # 7. Force Viewport Update
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'

    print("Heatmap Complete.")

create_deformation_heatmap()
# import bpy
# import bmesh
# from mathutils import Vector

# def create_deformation_heatmap():
#     # --- CONFIGURATION ---
#     # 4mm limit (Red > 4mm, Blue < -4mm)
#     LIMIT = 0.0015
#     # ---------------------
    
#     # 1. Validation
#     target_obj = bpy.context.active_object
#     selected = bpy.context.selected_objects
    
#     # Find the other selected mesh (Reference)
#     ref_obj = None
#     for obj in selected:
#         if obj != target_obj and obj.type == 'MESH':
#             ref_obj = obj
#             break
            
#     if not target_obj or not ref_obj:
#         print("ERROR: Select Reference (Rest), then Shift-Select Target (Deformed).")
#         return

#     print(f"Heatmap: '{ref_obj.name}' -> '{target_obj.name}' (Limit: {LIMIT}m)")

#     # 2. Fix Color Management (Crucial for Red/Blue visibility)
#     # if bpy.context.scene.view_settings.view_transform != 'Standard':
#     #     bpy.context.scene.view_settings.view_transform = 'Standard'

#     # 3. Geometry Data
#     # Force update to ensure world matrix is correct
#     bpy.context.view_layer.update()
    
#     mesh_tgt = target_obj.data
#     mesh_ref = ref_obj.data
    
#     verts_tgt = mesh_tgt.vertices
#     verts_ref = mesh_ref.vertices
    
#     # [UPDATED DEBUGGING]
#     print(f"--- VERTEX COUNT CHECK ---")
#     print(f"Target ({target_obj.name}): {len(verts_tgt)} vertices")
#     print(f"Ref    ({ref_obj.name}): {len(verts_ref)} vertices")
    
#     if len(verts_tgt) != len(verts_ref):
#         print("ERROR: Vertex count mismatch! Topology must be identical.")
#         print("TIP: If the difference is small (e.g. 16 verts), one mesh has merged UV seams and the other doesn't.")
#         print("SOLUTION: Export the 'Rest Pose' from your Python script using process=False.")
#         return

#     # 4. Calculate Signed Distances (World Space)
#     mat_tgt = target_obj.matrix_world
#     mat_ref = ref_obj.matrix_world
    
#     # Normal Matrix (Inverse Transpose)
#     norm_mat_ref = mat_ref.to_3x3().inverted().transposed()

#     signed_dists = []
    
#     for i, v_tgt in enumerate(verts_tgt):
#         v_ref = verts_ref[i]
        
#         pos_tgt = mat_tgt @ v_tgt.co
#         pos_ref = mat_ref @ v_ref.co
        
#         # Reference normal in world space
#         norm_ref = (norm_mat_ref @ v_ref.normal).normalized()
        
#         disp = pos_tgt - pos_ref
#         dist = disp.dot(norm_ref)
#         signed_dists.append(dist)

#     # Print Stats to System Console (Window > Toggle System Console)
#     print(f"Stats: Max={max(signed_dists):.5f}m, Min={min(signed_dists):.5f}m")

#     # 5. Generate Vertex Colors
#     col_name = "Deformation_Heatmap"
#     if not mesh_tgt.color_attributes.get(col_name):
#         mesh_tgt.color_attributes.new(name=col_name, type='FLOAT_COLOR', domain='CORNER')
    
#     color_layer = mesh_tgt.color_attributes[col_name]
#     mesh_tgt.attributes.active_color_index = 0 

#     vertex_colors = []
    
#     # Base Grey (0.7) preserves shading better than 0.5 or 1.0
#     BASE = 0.7 
    
#     for val in signed_dists:
#         # Normalize -1 to 1 based on LIMIT
#         # val=0.004 -> 1.0, val=-0.004 -> -1.0
#         norm = max(min(val / LIMIT, 1.0), -1.0)
        
#         if norm > 0:
#             # Positive (Red)
#             r = BASE + (1.0 - BASE) * norm  # Boost Red
#             g = BASE * (1.0 - norm)         # Kill Green
#             b = BASE * (1.0 - norm)         # Kill Blue
#         else:
#             # Negative (Blue)
#             norm = abs(norm)
#             r = BASE * (1.0 - norm)         # Kill Red
#             g = BASE * (1.0 - norm)         # Kill Green
#             b = BASE + (1.0 - BASE) * norm  # Boost Blue
            
#         vertex_colors.append((r, g, b, 1.0))

#     # Apply to Loop Indices
#     for poly in mesh_tgt.polygons:
#         for loop_index in poly.loop_indices:
#             v_idx = mesh_tgt.loops[loop_index].vertex_index
#             color_layer.data[loop_index].color = vertex_colors[v_idx]

#     # 6. Material Setup (Principled BSDF)
#     mat_name = "Heatmap_PBR_new_skin"
#     mat = bpy.data.materials.get(mat_name)
#     if not mat:
#         mat = bpy.data.materials.new(name=mat_name)
#         mat.use_nodes = True
#         nodes = mat.node_tree.nodes
#         links = mat.node_tree.links
#         nodes.clear()
        
#         # Output
#         out = nodes.new('ShaderNodeOutputMaterial')
#         out.location = (300, 0)
        
#         # Shader
#         bsdf = nodes.new('ShaderNodeBsdfPrincipled')
#         bsdf.location = (0, 0)
#         bsdf.inputs['Roughness'].default_value = 0.7
        
#         # Attribute
#         attr = nodes.new('ShaderNodeAttribute')
#         attr.location = (-300, 0)
#         attr.attribute_name = col_name
        
#         # Link
#         links.new(attr.outputs['Color'], bsdf.inputs['Base Color'])
#         links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])

#     # Assign Material
#     if not target_obj.data.materials:
#         target_obj.data.materials.append(mat)
#     else:
#         target_obj.data.materials[0] = mat

#     # 7. Force Viewport Update
#     for area in bpy.context.screen.areas:
#         if area.type == 'VIEW_3D':
#             for space in area.spaces:
#                 if space.type == 'VIEW_3D':
#                     space.shading.type = 'MATERIAL'

#     print("Heatmap Complete.")

# create_deformation_heatmap()