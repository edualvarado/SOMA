"""
Script: playback_precomputed_deformation.py
Goal: A lightweight script that reads a pre-computed deformation cache
      for extremely fast playback of complex muscle deformations.
"""
import bpy
import json
import numpy as np
from mathutils import Vector, Matrix

# --- User Configuration ---

# ---
shot = "shot_002"  # Change this to your shot name
FRAME_LIMIT = 200  # Limit to first 1000 frames for performance
# ---

# This is the final cache file created by the last script
DEFORMATION_CACHE_JSON_PATH = f"C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/displacements/{shot}/final_refined_deformation_cache_{shot}_{FRAME_LIMIT}.json"

# The collection containing your original, static A-pose muscle meshes
MUSCLE_COLLECTION_NAME = f"canonical_muscle_complex_{shot[-3:]}" # Collection containing ALL muscle objects

# Naming for the new objects and collection
DEFORMED_COLLECTION_NAME = "Final_Deformed_Muscles_Playback"

# --- Visualization Settings ---
EXAGGERATION_SCALE_DEFAULT = 1.0  # Start with 1.0 for true deformation, increase to see effect
# ----------------------------------------------------

# --- Global dictionary to hold playback data ---
PLAYBACK_CACHE = {}


def on_frame_change_playback(scene):
    """Extremely fast handler that only reads from cache and updates vertices."""
    # Do nothing if data isn't loaded or there are no objects to deform
    if not PLAYBACK_CACHE.get("pose_data") or not PLAYBACK_CACHE.get("deformed_objects"):
        return

    current_frame_str = str(scene.frame_current)
    # Get the dictionary of pre-computed vertex positions for this frame
    data_for_this_frame = PLAYBACK_CACHE["pose_data"].get(current_frame_str)

    # Get the user-controlled exaggeration scale from the UI
    scale_factor = scene.get("muscle_exaggeration_scale", 1.0)

    if data_for_this_frame:
        # Loop through the dictionary of objects we are controlling
        for muscle_name, deformed_obj in PLAYBACK_CACHE["deformed_objects"].items():

            # Robustness check: ensure the object wasn't deleted by the user
            if not deformed_obj or deformed_obj.name not in scene.objects:
                continue

            # Get the pre-computed vertex list for this muscle
            new_coords_list = data_for_this_frame.get(muscle_name)

            if new_coords_list:
                # Get the original A-pose positions for this muscle
                p_unposed = PLAYBACK_CACHE["p_unposed_muscles"][muscle_name]

                # --- Apply exaggeration ---
                # The cache stores final positions, so we find the displacement first
                displacements = np.array(new_coords_list, dtype=np.float32) - p_unposed
                # Add the scaled displacement to the original positions
                final_positions = p_unposed + (displacements * scale_factor)

                # Update mesh vertices in one fast operation
                deformed_obj.data.vertices.foreach_set("co", final_positions.ravel())
                deformed_obj.data.update()


def setup_playback():
    """Sets up the scene for fast playback from the pre-computed cache."""
    global PLAYBACK_CACHE
    print("--- MODE: Setting up Pre-computed Deformation Playback ---")

    # --- 1. Cleanup old visualization objects if they exist ---
    if DEFORMED_COLLECTION_NAME in bpy.data.collections:
        coll_to_delete = bpy.data.collections[DEFORMED_COLLECTION_NAME]
        for obj in list(coll_to_delete.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(coll_to_delete)

    # --- 2. Load the pre-computed cache file ---
    print(f"Loading deformation cache from {DEFORMATION_CACHE_JSON_PATH}...")
    try:
        with open(DEFORMATION_CACHE_JSON_PATH, 'r') as f:
            PLAYBACK_CACHE["pose_data"] = json.load(f)
    except Exception as e:
        print(f"ERROR loading cache JSON: {e}");
        return
    print("Cache loaded.")

    # --- 3. Get source muscles and create duplicates for visualization ---
    muscle_collection = bpy.data.collections.get(MUSCLE_COLLECTION_NAME)
    if not muscle_collection:
        print(f"ERROR: Source muscle collection '{MUSCLE_COLLECTION_NAME}' not found.");
        return
    source_muscle_objects = [obj for obj in muscle_collection.objects if obj.type == 'MESH']

    deformed_collection = bpy.data.collections.new(DEFORMED_COLLECTION_NAME)
    bpy.context.scene.collection.children.link(deformed_collection)

    PLAYBACK_CACHE["deformed_objects"] = {}
    PLAYBACK_CACHE["p_unposed_muscles"] = {}  # Cache A-pose positions for exaggeration

    print("Creating duplicate muscle objects for playback...")
    for source_obj in source_muscle_objects:
        deformed_obj = source_obj.copy()
        deformed_obj.data = source_obj.data.copy()
        deformed_obj.name = source_obj.name + "_playback"
        deformed_collection.objects.link(deformed_obj)

        # Store the object reference and its A-pose vertices in the cache
        PLAYBACK_CACHE["deformed_objects"][source_obj.name] = deformed_obj
        PLAYBACK_CACHE["p_unposed_muscles"][source_obj.name] = np.array(
            [source_obj.matrix_world @ v.co for v in source_obj.data.vertices], dtype=np.float32)

    # Hide the original muscles so we only see the deforming ones
    muscle_collection.hide_viewport = True
    print(f"Created {len(PLAYBACK_CACHE['deformed_objects'])} duplicate muscles for playback.")

    # --- 4. Setup Slider and Register the Animation Handler ---
    bpy.context.scene["muscle_exaggeration_scale"] = EXAGGERATION_SCALE_DEFAULT
    bpy.context.scene.id_properties_ui("muscle_exaggeration_scale").update(min=0.0, max=50.0, step=0.1,
                                                                           description="Exaggerate non-rigid muscle deformation")

    # Clear any old handlers and register our new, fast one
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(on_frame_change_playback)

    # Run once to set the initial pose
    on_frame_change_playback(bpy.context.scene)

    print("\n--- Playback Setup Complete ---")
    print("Handler registered. Play or scrub the timeline for fast, smooth playback.")
    print("Find the 'muscle_exaggeration_scale' slider in Scene Properties -> Custom Properties.")


# --- Run the main setup function ---
if __name__ == "__main__":
    setup_playback()