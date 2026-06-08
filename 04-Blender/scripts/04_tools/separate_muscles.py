import bpy

# --- User Configuration ---
# Name of the parent collection where the new folder will be created
# If this doesn't exist, it will be created in the Scene Root.
USER_PARENT_COLLECTION = "FBX-S5-TPose" 

def separate_muscles_preserving_original():
    # 1. Get the currently active object
    original_obj = bpy.context.active_object

    # Check if valid
    if not original_obj or original_obj.type != 'MESH':
        print("Error: Please select a Mesh object first.")
        return

    print(f"Processing: {original_obj.name} (Non-Destructive Mode)...")

    # 2. Setup Collections
    # Ensure User Parent Collection exists
    if USER_PARENT_COLLECTION in bpy.data.collections:
        parent_col = bpy.data.collections[USER_PARENT_COLLECTION]
    else:
        parent_col = bpy.data.collections.new(USER_PARENT_COLLECTION)
        bpy.context.scene.collection.children.link(parent_col)

    # Create the specific collection for these muscles
    new_col_name = f"{original_obj.name}_Separated"
    if new_col_name in bpy.data.collections:
        # Avoid duplicates if running multiple times
        new_muscle_col = bpy.data.collections[new_col_name]
    else:
        new_muscle_col = bpy.data.collections.new(new_col_name)
        parent_col.children.link(new_muscle_col)

    # 3. Duplicate the Object (So we don't touch the original)
    # Create a full copy of the object and its mesh data
    new_mesh = original_obj.data.copy()
    new_obj = original_obj.copy()
    new_obj.data = new_mesh
    new_obj.name = f"{original_obj.name}_Copy"

    # Link the new copy ONLY to the new collection
    new_muscle_col.objects.link(new_obj)

    # 4. Separate the Duplicate
    # Deselect everything, select only the new copy
    bpy.ops.object.select_all(action='DESELECT')
    new_obj.select_set(True)
    bpy.context.view_layer.objects.active = new_obj

    # Enter Edit Mode -> Select All -> Separate
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.separate(type='LOOSE')
    bpy.ops.object.mode_set(mode='OBJECT')

    # 5. Cleanup and Organize
    # Get all objects currently selected (these are the separated parts)
    separated_parts = bpy.context.selected_objects

    for part in separated_parts:
        # Reset Origin to Geometry for each part
        bpy.context.view_layer.objects.active = part
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        
        # Rename nicely (optional)
        # It usually defaults to ObjectName.001, ObjectName.002, etc.

    print(f"Success! Original kept safe. Created {len(separated_parts)} muscles in collection: '{USER_PARENT_COLLECTION} > {new_col_name}'")

# Execute
separate_muscles_preserving_original()