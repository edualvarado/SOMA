import bpy

# --- CONFIGURATION ---
# !!! IMPORTANT: Change these names to match your scene !!!
TARGET_OBJECT_NAME = "musc_mesh_complex"  # The name of the single, combined mesh object to separate.
# The collection where the new, separated parts will be placed.
RESULT_COLLECTION_NAME = "Muscles_Separated_Result"


def main():
    """Main function to run the script."""
    # --- 1. VALIDATION AND SETUP ---
    target_object = bpy.data.objects.get(TARGET_OBJECT_NAME)
    if not target_object:
        print(f"Error: Target object '{TARGET_OBJECT_NAME}' not found. Please check the name.")
        return
    if target_object.type != 'MESH':
        print(f"Error: Target object '{TARGET_OBJECT_NAME}' is not a mesh.")
        return

    # Create or get the result collection
    result_collection = bpy.data.collections.get(RESULT_COLLECTION_NAME)
    if not result_collection:
        result_collection = bpy.data.collections.new(RESULT_COLLECTION_NAME)
        bpy.context.scene.collection.children.link(result_collection)

    print(f"Target object to separate is '{TARGET_OBJECT_NAME}'.")
    print("Starting separation by loose parts...")

    # --- 2. SEPARATION LOGIC ---
    # We need to be in Object Mode to safely select the object
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # Deselect all objects to ensure we only select our target
    bpy.ops.object.select_all(action='DESELECT')

    # Select and activate the target object
    bpy.context.view_layer.objects.active = target_object
    target_object.select_set(True)

    # --- ROBUST OBJECT IDENTIFICATION ---
    # Get the set of all mesh objects before separating
    objects_before = {obj for obj in bpy.data.objects if obj.type == 'MESH'}

    # Switch to Edit Mode and separate by loose parts
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.separate(type='LOOSE')

    # Switch back to Object mode to evaluate the result
    bpy.ops.object.mode_set(mode='OBJECT')

    # Get the set of all mesh objects after separating
    objects_after = {obj for obj in bpy.data.objects if obj.type == 'MESH'}

    # The new objects are the difference between the two sets
    new_objects = objects_after - objects_before

    # --- 3. ORGANIZE NEW OBJECTS ---
    if new_objects:
        print(f"Successfully separated into {len(new_objects)} new objects.")
        for new_obj in new_objects:
            # Unlink the new object from its current collection (likely the scene's root)
            for coll in new_obj.users_collection:
                coll.objects.unlink(new_obj)
            # Link it to our result collection
            result_collection.objects.link(new_obj)
        print(f"All new parts have been moved to the '{RESULT_COLLECTION_NAME}' collection.")
    else:
        print("Separation did not result in any new objects.")

    # --- 4. CLEANUP ---
    # The original object should now be empty. We can delete it.
    # Check if the original target object still exists and if it's empty
    if target_object and len(target_object.data.vertices) == 0:
        print(f"Original target '{target_object.name}' is now empty and will be deleted.")
        bpy.data.objects.remove(target_object, do_unlink=True)
    else:
        print(f"Original target '{target_object.name}' still contains some geometry.")

    print(f"\n--- Script Finished ---")


# --- RUN THE SCRIPT ---
if __name__ == "__main__":
    main()