"""
Script: export_lbs_weights_skin.py
Goal: Takes the 3D model in Blender and exports the LBS weights in a JSON file like 'skin_lbs_weights_exported.json'.
"""

import bpy
import json
import time  # For tracking time

def is_finger_bone(bone_name):
    """Check if a bone is a finger bone that should be excluded."""
    finger_patterns = [
        "thumb_01", "thumb_02", "thumb_03",
        "index_01", "index_02", "index_03",
        "middle_01", "middle_02", "middle_03",
        "ring_01", "ring_02", "ring_03",
        "pinky_01", "pinky_02", "pinky_03"
    ]
    
    bone_name_lower = bone_name.lower()
    for pattern in finger_patterns:
        # Check if pattern is in the bone name (handles _L, _R, -L, -R suffixes)
        if pattern in bone_name_lower:
            return True
    return False

# --- USER: SET THESE ---

# SKIN FOR THE NEW SKELETON (INSIDE HUMANS ADAPTED TO STUDIO)
# SKINNED_OBJECT_NAME = "skin-NewSkeleton"
# SKINNED_OBJECT_NAME = "skin-NewSkeleton-TPose"
# SKINNED_OBJECT_NAME = "canonical_muscle_complex_fbx_unified-NewSkeleton"

# SKINNED_OBJECT_NAME = "skin_layer-S1"

SKINNED_OBJECT_NAME = "skin_layer-S1-APose"
# SKINNED_OBJECT_NAME = "musc_layer-S4-APose"
# SKINNED_OBJECT_NAME = "musc_meshes_unified-S5-APose"

# SKIN WEIGHTS FOR THE NEW SKELETON (INSIDE HUMANS ADAPTED TO STUDIO)
# OUTPUT_WEIGHTS_FILE = "S:/work/03-MUSK/04-Blender/data/weights/canonical_model/lbs_skin/skin_lbs_weights_exported_new_skeleton.json"
# OUTPUT_WEIGHTS_FILE = "S:/work/03-MUSK/04-Blender/data/weights/canonical_model/lbs_skin/skin_lbs_weights_exported_new_skeleton_tpose.json"
# OUTPUT_WEIGHTS_FILE = "S:/work/03-MUSK/04-Blender/data/weights/canonical_model/lbs_skin/musc_meshes_lbs_weights_exported_new_skeleton_tpose.json"

# OUTPUT_WEIGHTS_FILE = "S:/work/03-MUSK/04-Blender/data/weights/canonical_model/lbs_skin/skin_lbs_weights_exported_tpose_NEW.json"

OUTPUT_WEIGHTS_FILE = "S:/work/03-MUSK/02-Canonical-Model/S1/weights/canonical_model/lbs_skin/S1_skin_lbs_weights_exported.json"
# OUTPUT_WEIGHTS_FILE = "S:/work/03-MUSK/02-Canonical-Model/S4/weights/canonical_model/lbs_skin/S4_musc_lbs_weights_exported.json"
# OUTPUT_WEIGHTS_FILE = "S:/work/03-MUSK/02-Canonical-Model/S5/weights/canonical_model/lbs_skin/S5_musc_meshes_unified_lbs_weights_exported.json"

# SKIN FOR THE OLD SKELETON (INSIDE HUMANS)
# SKINNED_OBJECT_NAME = "skin-Original"

# SKIN WEIGHTS FOR THE OLD SKELETON (INSIDE HUMANS)
# OUTPUT_WEIGHTS_FILE = "S:/work/03-MUSK/04-Blender/data/weights/canonical_model/lbs_skin/skin_lbs_weights_exported_original.json"

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
    mesh = obj.data

    vertex_weights_data = {}
    exported_vertex_data_list = []

    # --- Bone Index Mapping ---
    armature_obj = None
    if obj.parent and obj.parent.type == 'ARMATURE':
        armature_obj = obj.parent
    else:  # Check modifiers
        for mod in obj.modifiers:
            if mod.type == 'ARMATURE' and mod.object:
                armature_obj = mod.object
                break

    bone_name_to_final_idx = {}
    finger_bones = []

    if armature_obj:
        print(f"Using armature '{armature_obj.name}' for bone indexing order.")
        final_idx = 0
        for bone in armature_obj.data.bones:
            bone_name_to_final_idx[bone.name] = final_idx
            # if is_finger_bone(bone.name):
            #     finger_bones.append(bone.name)
            #     print(f"  Finger bone (will have 0 weight): {bone.name}")
            final_idx += 1
    else:
        print("Warning: No clear Armature parent or modifier found.")
        final_idx = 0
        for vg in obj.vertex_groups:
            bone_name_to_final_idx[vg.name] = final_idx
            # if is_finger_bone(vg.name):
            #     finger_bones.append(vg.name)
            #     print(f"  Finger bone (will have 0 weight): {vg.name}")
            final_idx += 1

    print(f"Total finger bones to zero out: {len(finger_bones)}")

    # Determine fallback bones for vertices that only have finger weights
    global_fallback_bone_idx = -1
    global_fallback_bone_name = "UNKNOWN_GLOBAL_FALLBACK"
    hand_R_idx = -1
    hand_L_idx = -1

    if armature_obj and armature_obj.data.bones:
        # Find global fallback (e.g., Hips)
        for i, bone in enumerate(armature_obj.data.bones):
            if bone.name.lower() in ["hips", "pelvis", "root"]: # Common root bone names
                global_fallback_bone_idx = bone_name_to_final_idx.get(bone.name, i) 
                global_fallback_bone_name = bone.name
                break
        if global_fallback_bone_idx == -1: # If no specific root found, just use the first bone
            global_fallback_bone_idx = bone_name_to_final_idx.get(armature_obj.data.bones[0].name, 0)
            global_fallback_bone_name = armature_obj.data.bones[0].name
        
        # Find hand bones for local fallback
        if "hand_R" in bone_name_to_final_idx:
            hand_R_idx = bone_name_to_final_idx["hand_R"]
        if "hand_L" in bone_name_to_final_idx:
            hand_L_idx = bone_name_to_final_idx["hand_L"]
    
    if global_fallback_bone_idx == -1:
        print("ERROR: Could not determine a global fallback bone. Exiting.")
        return

    print(f"Global fallback bone: '{global_fallback_bone_name}' (index: {global_fallback_bone_idx})")
    print(f"Hand_R fallback bone index: {hand_R_idx}")
    print(f"Hand_L fallback bone index: {hand_L_idx}")


    # --- Vertex Iteration ---
    num_total_vertices = len(mesh.vertices)
    print(f"Total vertices to process: {num_total_vertices}")

    if num_total_vertices == 0:
        print("ERROR: Mesh has no vertices.")
        return

    next_log_percent = LOG_PERCENT_INTERVAL
    if LOG_PERCENT_INTERVAL <= 0:
        next_log_percent = 101

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
            while next_log_percent <= current_percent:
                next_log_percent += LOG_PERCENT_INTERVAL
        # --- End Progress Logging ---

        # Collect all weights, zeroing out finger bone weights
        temp_weights = []
        temp_bone_indices = []
        temp_bone_names = []

        for group_element in vertex.groups:
            group_idx = group_element.group
            weight = group_element.weight

            if weight > 1e-6: # Only consider weights above a threshold
                vertex_group_name = obj.vertex_groups[group_idx].name
                final_bone_idx = bone_name_to_final_idx.get(vertex_group_name)

                if final_bone_idx is not None:
                    # Set finger bone weights to 0.0
                    # if is_finger_bone(vertex_group_name):
                    #     weight = 0.0
                    
                    temp_bone_indices.append(final_bone_idx)
                    temp_weights.append(weight)
                    temp_bone_names.append(vertex_group_name)
                else:
                    print(
                        f"Warning: Vertex group '{vertex_group_name}' on v_idx {v_idx} not found in bone_name_to_final_idx map. Skipping this weight.")

        # Now, filter out any bones that ended up with 0.0 weight and normalize the rest
        final_weights = []
        final_bone_indices = []
        final_bone_names = []

        for i in range(len(temp_weights)):
            if temp_weights[i] > 1e-6: # Only keep non-zero weights
                final_weights.append(temp_weights[i])
                final_bone_indices.append(temp_bone_indices[i])
                final_bone_names.append(temp_bone_names[i])

        if final_weights:
            # Normalize the remaining non-zero weights
            sum_of_weights = sum(final_weights)
            if sum_of_weights > 1e-6:
                normalized_weights = [w / sum_of_weights for w in final_weights]
            else:
                # This case should ideally not happen if final_weights is not empty and sum > 1e-6
                # But as a safeguard, if it does, assign to appropriate hand bone
                print(f"Warning: Vertex {v_idx} had non-zero weights that summed to zero after filtering. Assigning to local fallback bone.")
                normalized_weights = [1.0]
                
                # Determine local fallback for this specific vertex
                local_fallback_bone_idx = global_fallback_bone_idx
                local_fallback_bone_name = global_fallback_bone_name
                
                found_side = None
                for bone_name in temp_bone_names: # Check original bone names to infer side
                    if "_R" in bone_name:
                        found_side = "R"
                        break
                    elif "_L" in bone_name:
                        found_side = "L"
                        break
                
                if found_side == "R" and hand_R_idx != -1:
                    local_fallback_bone_idx = hand_R_idx
                    local_fallback_bone_name = "hand_R"
                elif found_side == "L" and hand_L_idx != -1:
                    local_fallback_bone_idx = hand_L_idx
                    local_fallback_bone_name = "hand_L"

                final_bone_indices = [local_fallback_bone_idx]
                final_bone_names = [local_fallback_bone_name]
        else:
            # This vertex had only finger bone weights (or all weights were effectively zero)
            # Assign it entirely to the appropriate hand bone
            normalized_weights = [1.0]
            
            # Determine local fallback for this specific vertex
            local_fallback_bone_idx = global_fallback_bone_idx
            local_fallback_bone_name = global_fallback_bone_name
            
            found_side = None
            for bone_name in temp_bone_names: # Check original bone names to infer side
                if "_R" in bone_name:
                    found_side = "R"
                    break
                elif "_L" in bone_name:
                    found_side = "L"
                    break
            
            if found_side == "R" and hand_R_idx != -1:
                local_fallback_bone_idx = hand_R_idx
                local_fallback_bone_name = "hand_R"
            elif found_side == "L" and hand_L_idx != -1:
                local_fallback_bone_idx = hand_L_idx
                local_fallback_bone_name = "hand_L"

            final_bone_indices = [local_fallback_bone_idx]
            final_bone_names = [local_fallback_bone_name]

        # Use final_bone_indices, final_bone_names, normalized_weights for export
        vx, vy, vz = vertex.co.x, vertex.co.y, vertex.co.z

        vertex_weights_data[v_idx] = {
            "bone_indices": final_bone_indices,
            "bone_names": final_bone_names,
            "weights": normalized_weights
        }

        exported_vertex_data_list.append({
            "blender_v_idx": v_idx,
            "coords_local": [vx, vy, vz],
            "bone_indices": final_bone_indices,
            "bone_names": final_bone_names,
            "weights": normalized_weights
        })

    total_time = time.time() - start_time
    print(f"Finished processing all {num_total_vertices} vertices. Total time: {total_time:.2f} seconds.")

    # --- Save to JSON ---
    try:
        with open(OUTPUT_WEIGHTS_FILE, 'w') as f:
            json.dump(exported_vertex_data_list, f, indent=2)
        print(
            f"Successfully exported skin weights for {len(exported_vertex_data_list)} effective vertices to {OUTPUT_WEIGHTS_FILE}")
    except IOError as e:
        print(f"ERROR writing output JSON to '{OUTPUT_WEIGHTS_FILE}'. Check path and permissions. Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while writing JSON: {e}")


if __name__ == "__main__":
    export_skin_weights()