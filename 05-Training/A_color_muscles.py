import bpy

# Ensure we are in Object Mode
if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')

selected_meshes = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
selected_meshes.sort(key=lambda x: x.name)

print(f"--- Assigning IDs to {len(selected_meshes)} muscles ---")

for i, obj in enumerate(selected_meshes):
    try:
        mesh = obj.data
        color_val = (i + 1) / 255.0
        
        layer_name = "ID_Color"
        
        # 1. Clean up existing layer if it exists to prevent conflicts
        if hasattr(mesh, "color_attributes"): # Blender 3.2+
            if layer_name in mesh.color_attributes:
                mesh.color_attributes.remove(mesh.color_attributes[layer_name])
            # Create fresh layer (FLOAT_COLOR ensures math precision, POINT = per-vertex)
            color_attr = mesh.color_attributes.new(name=layer_name, type='FLOAT_COLOR', domain='POINT')
            
            # 2. Fast assignment: Create a flat list of RGBA values for all vertices
            # e.g., [R, G, B, A, R, G, B, A, ...]
            num_verts = len(mesh.vertices)

            # colors_flat = [color_val, 0.0, 0.0, 1.0] * num_verts
            
            import colorsys
            n = max(1, len(selected_meshes))
            hue = i / n
            r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
            colors_flat = [r, g, b, 1.0] * num_verts

            # Apply all colors at once (much safer and faster)
            color_attr.data.foreach_set("color", colors_flat)
            
            # Make it the active render layer
            mesh.color_attributes.active_color = color_attr

        else: # Legacy Blender (Pre-3.2)
            if layer_name in mesh.vertex_colors:
                mesh.vertex_colors.remove(mesh.vertex_colors[layer_name])
            color_attr = mesh.vertex_colors.new(name=layer_name)
            
            rgba = (color_val, 0.0, 0.0, 1.0)
            for loop in mesh.loops:
                color_attr.data[loop.index].color = rgba

        print(f"Assigned ID {i+1:02d} | R={color_val:.4f} | to {obj.name}")

    except Exception as e:
        # IF IT FAILS, THIS WILL TELL US EXACTLY WHY
        print(f"❌ ERROR on {obj.name}: {e}")

print("--- Done! ---")