import bpy

def create_live_cross_section():
    # 1. Get the target object
    target_obj = bpy.context.active_object
    
    if not target_obj or target_obj.type != 'MESH':
        print("Error: Please select a Mesh object.")
        return

    # 2. Create the Cutter Object (Cube)
    # We create it at the target's location
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=target_obj.location)
    cutter_obj = bpy.context.active_object
    cutter_obj.name = f"Cutter_for_{target_obj.name}"
    
    # 3. Setup Cutter Visuals
    # Set to 'BOUNDS' so it looks like a wireframe box and doesn't block the view
    cutter_obj.display_type = 'BOUNDS'
    cutter_obj.hide_render = True  # Invisible in final render

    # 4. Add the Boolean Modifier to the Target
    # We must make the target active again to add modifiers
    bpy.context.view_layer.objects.active = target_obj
    
    mod = target_obj.modifiers.new(name="Live_Cross_Section", type='BOOLEAN')
    mod.object = cutter_obj
    mod.operation = 'DIFFERENCE'
    mod.solver = 'FAST'  # 'FAST' is better for real-time dragging

    # 5. Select the Cutter automatically for the user
    # Deselect everything
    bpy.ops.object.select_all(action='DESELECT')
    # Select the cutter and make it active
    cutter_obj.select_set(True)
    bpy.context.view_layer.objects.active = cutter_obj

    print("Success! Move the selected box to cut the mesh.")

create_live_cross_section()