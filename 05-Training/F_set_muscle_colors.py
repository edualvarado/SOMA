import bpy
import bmesh
import numpy as np

# --- CONFIGURATION ---
TARGET_OBJ_NAME = bpy.context.active_object.name
LAYER_NAME = "ID_Color"
# ---------------------

def create_id_material(obj, attribute_name):
    """Auto-generates a material that displays the Vertex Color attribute"""
    mat_name = "Muscle_ID_Material"
    
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    node_out = nodes.new('ShaderNodeOutputMaterial')
    node_out.location = (300, 0)
    
    node_bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    node_bsdf.location = (0, 0)
    
    # Node to read the vertex colors
    node_attr = nodes.new('ShaderNodeAttribute') 
    node_attr.location = (-300, 0)
    node_attr.attribute_type = 'GEOMETRY'
    node_attr.attribute_name = attribute_name
    
    links.new(node_attr.outputs['Color'], node_bsdf.inputs['Base Color'])
    links.new(node_bsdf.outputs['BSDF'], node_out.inputs['Surface'])
    
    # Assign to object
    if len(obj.data.materials) == 0:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat

def main():
    obj = bpy.data.objects.get(TARGET_OBJ_NAME)
    if not obj or obj.type != 'MESH':
        print("Please select a mesh object.")
        return

    mesh = obj.data
    
    # 1. Clean up old/conflicting color attributes from OBJ import
    while mesh.color_attributes:
        mesh.color_attributes.remove(mesh.color_attributes[0])

    # 2. Find islands (loose parts)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    
    islands_indices = []
    undiscovered = set(bm.verts)
    
    while undiscovered:
        v = undiscovered.pop()
        # Store the integer index right away, not the BMVert object
        island = {v.index} 
        queue = [v]
        while queue:
            current_v = queue.pop()
            for edge in current_v.link_edges:
                v2 = edge.other_vert(current_v)
                if v2 in undiscovered:
                    undiscovered.remove(v2)
                    island.add(v2.index)
                    queue.append(v2)
        islands_indices.append(list(island))

    print(f"Found {len(islands_indices)} individual muscle islands.")
    
    # NOW it is safe to free the BMesh!
    bm.free()

    # 3. Setup New Color Attribute
    color_attr = mesh.color_attributes.new(name=LAYER_NAME, type='FLOAT_COLOR', domain='POINT')
    mesh.color_attributes.active_color = color_attr
    
    colors = np.ones((len(mesh.vertices), 4), dtype=np.float32)

    # 4. Assign Groups and Colors
    for i, v_indices in enumerate(islands_indices):
        m_name = f"Muscle_{i:03d}"
        
        # Create Vertex Group
        if m_name not in obj.vertex_groups:
            vg = obj.vertex_groups.new(name=m_name)
        else:
            vg = obj.vertex_groups[m_name]
            
        vg.add(v_indices, 1.0, 'REPLACE')

        # Generate Random Color
        np.random.seed(i)
        random_color = np.random.rand(3)
        for v_idx in v_indices:
            colors[v_idx, :3] = random_color

    # Apply data back to mesh
    color_attr.data.foreach_set("color", colors.flatten())
    mesh.update()

    # 5. Apply the Material
    create_id_material(obj, LAYER_NAME)

    print(f"[SUCCESS] {len(islands_indices)} muscles colored and grouped!")

main()