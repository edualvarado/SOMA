"""
Script: export_lbs_weights_skin.py
Goal: Takes the 3D model in Blender and exports the LBS weights in a JSON file like 'skin_lbs_weights_exported.json'.
"""

import bpy
import json
import time  # For tracking time

# --- USER: SET THESE ---
# Exact name of your skinned suit mesh object in Blender's outliner
SKINNED_OBJECT_NAME = "canonical_muscle_complex_fbx_unified"


# Full path where the output JSON file will be saved
# OUTPUT_WEIGHTS_FILE = "S:/work/03-MUSK/04-Blender/data/weights/canonical_model/lbs_musc/wrap_musc_lbs_weights_exported_tpose.json"
OUTPUT_WEIGHTS_FILE = "S:/work/03-MUSK/04-Blender/data/weights/canonical_model/lbs_musc/muscle_complex_unified_lbs_weights_exported_tpose_new.json"

# ---------------------

# --- Logging Configuration ---
# Log progress every 'LOG_PERCENT_INTERVAL' percent. Set to 0 to disable.
LOG_PERCENT_INTERVAL = 5  # e.g., 5 means log at 5%, 10%, 15%...
# Alternatively, or additionally, log every N vertices. Set to 0 to disable.
LOG_EVERY_N_VERTICES = 1000  # e.g., log every 1000 vertices

def export_skin_weights():
    obj = bpy.data.objects.get(SKINNED_OBJECT_NAME)

    if not obj:
        print(f"ERROR: Object '{SKINNED_OBJECT_NAME}' not found in the scene.")
        return
    if obj.type != 'MESH':
        print(f"ERROR: Object '{SKINNED_OBJECT_NAME}' is not a MESH object (it's a {obj.type}).")
        return
    if not obj.vertex_groups:
        print(f"ERROR: Object '{SKINNED_OBJECT_NAME}' has no vertex groups (not skinned?).")
        return

    print(f"Processing skin weights for object: '{obj.name}'")


    # mesh = obj.data
    # =================================================================================
    # FIX #1: Get the EVALUATED mesh, which includes all modifiers (like Subdivision).
    # This ensures we get the final, high-resolution vertex count.
    # =================================================================================
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_obj = obj.evaluated_get(depsgraph)
    mesh = evaluated_obj.data
    # =================================================================================



    # We change this:
    vertex_weights_data = {}  # To store data: {vertex_index_int: {"bone_indices": [...], "weights": [...]}}
    exported_vertex_data_list = []


    # --- Bone Index Mapping ---
    # We need a consistent way to map bone names (from vertex groups) to integer indices for LBS.
    # Option 1: If the object has an Armature modifier and is parented to an Armature
    armature_obj = None
    if obj.parent and obj.parent.type == 'ARMATURE':
        armature_obj = obj.parent
    else:  # Check modifiers
        for mod in obj.modifiers:
            if mod.type == 'ARMATURE' and mod.object:
                armature_obj = mod.object
                break

    bone_name_to_final_idx = {}
    if armature_obj:
        print(f"Using armature '{armature_obj.name}' for bone indexing order.")
        # This ensures indices match the armature's bone list order
        for idx, bone in enumerate(armature_obj.data.bones):  # Iterates actual bones in armature
            bone_name_to_final_idx[bone.name] = idx
    else:
        print("Warning: No clear Armature parent or modifier found.")
        print("Bone indices will be based on vertex group order. Ensure this is consistent with your skeleton.")
        # Fallback: Use vertex group index if names are not critical or map directly
        for idx, vg in enumerate(obj.vertex_groups):
            bone_name_to_final_idx[vg.name] = idx  # Bone name maps to its vertex group index

    # --- Vertex Iteration ---
    num_total_vertices = len(mesh.vertices)
    print(f"Total vertices to process: {num_total_vertices}")

    if num_total_vertices == 0:
        print("ERROR: Mesh has no vertices.")
        return

    next_log_percent = LOG_PERCENT_INTERVAL
    if LOG_PERCENT_INTERVAL <= 0:  # Ensure it starts if interval is 0 or less
        next_log_percent = 101  # Effectively disable percent logging if not positive

    start_time = time.time()

    for v_idx, vertex in enumerate(mesh.vertices):
        # --- Progress Logging ---
        processed_count = v_idx + 1
        if LOG_EVERY_N_VERTICES > 0 and processed_count % LOG_EVERY_N_VERTICES == 0:
            elapsed_time = time.time() - start_time
            print(f"  Processed {processed_count}/{num_total_vertices} vertices... ({elapsed_time:.2f}s elapsed)")

        current_percent = (processed_count / num_total_vertices) * 100
        if current_percent >= next_log_percent:
            elapsed_time = time.time() - start_time
            print(
                f"  Progress: {int(next_log_percent)}% ({processed_count}/{num_total_vertices} vertices). Time: {elapsed_time:.2f}s")
            while next_log_percent <= current_percent:  # Handle cases where jumps over multiple intervals
                next_log_percent += LOG_PERCENT_INTERVAL
        # --- End Progress Logging ---

        weights_for_this_vertex = []
        bone_indices_for_this_vertex = []
        bone_names_for_this_vertex = []

        for group_element in vertex.groups:  # Iterate through vertex groups this vertex belongs to
            group_idx = group_element.group  # This is the index of the vertex group in obj.vertex_groups
            weight = group_element.weight

            if weight > 1e-6:  # Only consider significant weights
                vertex_group_name = obj.vertex_groups[group_idx].name  # Name of the bone/group

                final_bone_idx = bone_name_to_final_idx.get(vertex_group_name)

                if final_bone_idx is not None:
                    bone_indices_for_this_vertex.append(final_bone_idx)
                    weights_for_this_vertex.append(weight)
                    bone_names_for_this_vertex.append(vertex_group_name)  # Optional
                else:
                    print(
                        f"Warning: Vertex group '{vertex_group_name}' on v_idx {v_idx} not found in bone_name_to_final_idx map. Skipping this weight.")

        if weights_for_this_vertex:
            # Normalize weights for this vertex (important as Blender allows sums > 1 sometimes)
            sum_of_weights = sum(weights_for_this_vertex)
            if sum_of_weights > 1e-6:
                normalized_weights = [w / sum_of_weights for w in weights_for_this_vertex]

                # Get vertex coordinates (local to the object)
                vx, vy, vz = vertex.co.x, vertex.co.y, vertex.co.z

                # Store data keyed by integer vertex index
                vertex_weights_data[v_idx] = {
                    "bone_indices": bone_indices_for_this_vertex,
                    "bone_names": bone_names_for_this_vertex, # Optional
                    "weights": normalized_weights
                }

                exported_vertex_data_list.append({
                    "blender_v_idx": v_idx, # Keep for debugging if needed
                    # "coords_local": [vx, vy, vz], # Store local coordinates
                    "coords_local": [vx, vz, -vy], # Store local coordinates <-- IMPORTANT! COORDINATE SYSTEM
                    "bone_indices": bone_indices_for_this_vertex,
                    "bone_names": bone_names_for_this_vertex, # Optional
                    "weights": normalized_weights
                })

            # else: All weights for this vertex were effectively zero.
            # Implicitly, it will have no entry or empty lists.

    total_time = time.time() - start_time
    print(f"Finished processing all {num_total_vertices} vertices. Total time: {total_time:.2f} seconds.")

    # --- Save to JSON ---
    try:
        with open(OUTPUT_WEIGHTS_FILE, 'w') as f:
            json.dump(exported_vertex_data_list, f, indent=2)  # indent for readability
        print(
            f"Successfully exported skin weights for {len(exported_vertex_data_list)} effective vertices to {OUTPUT_WEIGHTS_FILE}")
    except IOError as e:
        print(f"ERROR writing output JSON to '{OUTPUT_WEIGHTS_FILE}'. Check path and permissions. Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while writing JSON: {e}")


if __name__ == "__main__":
    # Ensure you set SKINNED_OBJECT_NAME and OUTPUT_WEIGHTS_FILE at the top
    export_skin_weights()