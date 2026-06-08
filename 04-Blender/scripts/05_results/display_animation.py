import bpy
import os
import re

# --- CONFIGURATION: MULTIPLE SEQUENCES ---
# Add as many blocks as you need.
# Each block creates a separate object in Blender.
DIRECTORIES_TO_LOAD = [
    # {
    #     "path": r"T:\static00\S1\exported_skin_meshes_0_300_shot_001\S1\shot_001_captury",
    #     "name": "exported_skin_meshes_0_300_shot_001",       # Name of the object in Blender
    #     "prefix": "s_final_frame_"    # File prefix to look for
    # },
    # {
    #     "path": r"T:\static00\S1\exported_musc_meshes_1000_1300_shot_001\S1\shot_001_captury",
    #     "name": "exported_musc_meshes_1000_1300_shot_001",       # Name of the object in Blender
    #     "prefix": "m_final_frame_"    # File prefix to look for
    # },
    # {
    #     "path": r"T:\static00\S1\exported_skin_meshes_1700_2000_shot_001\S1\shot_001_captury",
    #     "name": "exported_skin_meshes_1700_2000_shot_001",
    #     "prefix": "s_final_frame_"
    # },
    # {
    #     "path": r"T:\static00\S1\exported_skin_meshes_50_350_shot_002\S1\shot_002_captury",
    #     "name": "exported_skin_meshes_50_350_shot_002",
    #     "prefix": "s_final_frame_"
    # },
    # {
    #     "path": r"T:\static00\S1\exported_skin_meshes_100_400_shot_003\S1\shot_003_captury",
    #     "name": "exported_skin_meshes_100_400_shot_003",
    #     "prefix": "s_final_frame_"
    # },
    {
        "path": r"T:\static00\S1\exported_musc_meshes_40_340_shot_004\S1\shot_004_captury",
        "name": "exported_musc_meshes_40_340_shot_004",
        "prefix": "m_final_frame_"
    }
]

FILE_EXTENSION = ".obj"
# ------------------------------------------

# Storage Format: 
# { "ObjectName": { 1000: mesh_data_1000, 1001: mesh_data_1001 } }
if "GLOBAL_FRAME_MAP" not in globals():
    GLOBAL_FRAME_MAP = {}

def extract_frame_number(filename):
    # Extracts the last number from a string
    # "m_final_frame_1000.obj" -> 1000
    nums = re.findall(r'\d+', filename)
    if nums:
        return int(nums[-1]) # Return the last number group found
    return None

def load_sequences_absolute():
    global GLOBAL_FRAME_MAP
    GLOBAL_FRAME_MAP.clear()

    window = bpy.context.window_manager.windows[0]
    with bpy.context.temp_override(window=window):
        
        for entry in DIRECTORIES_TO_LOAD:
            folder_path = entry["path"]
            obj_name = entry["name"]
            prefix = entry["prefix"]
            
            if not os.path.exists(folder_path):
                print(f"[SKIP] Not found: {folder_path}")
                continue

            print(f"--- Processing: {obj_name} ---")
            
            # Find files
            files = [f for f in os.listdir(folder_path) 
                     if f.startswith(prefix) and f.endswith(FILE_EXTENSION)]
            
            if not files:
                print(f"[SKIP] No matching files in {folder_path}")
                continue

            print(f"Loading {len(files)} frames for {obj_name}...")
            
            # Initialize dictionary for this object
            frame_dict = {} # { frame_number: mesh_data }
            
            first_mesh = None

            for fname in files:
                full_path = os.path.join(folder_path, fname)
                
                # 1. Determine exact frame number from filename
                frame_num = extract_frame_number(fname)
                if frame_num is None:
                    print(f"Could not parse number from: {fname}")
                    continue

                try:
                    # 2. Import
                    if bpy.app.version >= (4, 0, 0):
                        bpy.ops.wm.obj_import(filepath=full_path)
                    else:
                        bpy.ops.import_scene.obj(filepath=full_path, use_split_objects=False)

                    if not bpy.context.selected_objects: continue
                        
                    obj = bpy.context.selected_objects[0]
                    # Name mesh data clearly: "S1_Container_1000"
                    obj.data.name = f"{obj_name}_{frame_num}" 
                    
                    # 3. Store in Dictionary
                    frame_dict[frame_num] = obj.data
                    
                    if first_mesh is None: first_mesh = obj.data
                    
                    bpy.data.objects.remove(obj, do_unlink=True)
                    
                except Exception as e:
                    print(f"Error loading {fname}: {e}")
                    continue
            
            if not frame_dict: continue

            # Store in global cache
            GLOBAL_FRAME_MAP[obj_name] = frame_dict

            # Create Container Object (using the first loaded mesh as default)
            if obj_name in bpy.data.objects:
                container = bpy.data.objects[obj_name]
                container.data = first_mesh
            else:
                container = bpy.data.objects.new(obj_name, first_mesh)
                bpy.context.collection.objects.link(container)
            
            print(f"[SUCCESS] Loaded {obj_name} with frames: {min(frame_dict.keys())} to {max(frame_dict.keys())}")

    # --- DEFINE HANDLER ---
    def absolute_sync_handler(scene):
        current_frame = scene.frame_current
        
        # Loop through all objects we loaded
        for obj_name, frame_map in GLOBAL_FRAME_MAP.items():
            obj = bpy.data.objects.get(obj_name)
            
            if obj:
                # DOES THIS EXACT FRAME EXIST?
                if current_frame in frame_map:
                    # Yes -> Swap geometry
                    if obj.data != frame_map[current_frame]:
                        obj.data = frame_map[current_frame]
                        # Optional: Ensure it's visible
                        # obj.hide_viewport = False
                else:
                    # No -> We are in a gap (e.g., frame 500)
                    # Options:
                    # 1. Do nothing (keeps last shape)
                    # 2. Hide object (requires scaling to 0 usually to avoid dependency cycles)
                    pass

    # Register
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(absolute_sync_handler)
    
    print(f"Absolute Sync Handler Registered!")

if __name__ == "__main__":
    load_sequences_absolute()

# ---

# import bpy
# import os
# import re
# from pathlib import Path

# # --- CONFIGURATION ---
# # Ensure this path is correct and uses forward slashes or raw string (r"...")
# OBJ_FOLDER = r"T:\static00\S1\exported_musc_meshes_0_300_shot_001\S1\shot_001_captury"
# FILE_PREFIX = "m_final_frame_"
# FILE_EXTENSION = ".obj"
# SEQUENCE_OBJ_NAME = "exported_musc_meshes_0_300_shot_001"
# # ---------------------

# def natural_sort_key(s):
#     return [int(text) if text.isdigit() else text.lower()
#             for text in re.split('([0-9]+)', s)]

# def load_tpose_sequence():
#     if not os.path.exists(OBJ_FOLDER):
#         print(f"Error: {OBJ_FOLDER} not found")
#         return

#     files = [f for f in os.listdir(OBJ_FOLDER) 
#              if f.startswith(FILE_PREFIX) and f.endswith(FILE_EXTENSION)]
#     files.sort(key=natural_sort_key)
    
#     if not files: 
#         print("No files found!")
#         return

#     print(f"Loading {len(files)} T-Pose frames...")
#     loaded_meshes = []
    
#     # [FIX] Get context override correctly
#     # We need a window and screen context for operators to work
#     window = bpy.context.window_manager.windows[0]
#     with bpy.context.temp_override(window=window):
        
#         for i, fname in enumerate(files):
#             path = os.path.join(OBJ_FOLDER, fname)
#             print(f"Loading {i}: {fname}")
            
#             try:
#                 # [FIX] Blender 4.0+ Compatibility Check
#                 if bpy.app.version >= (4, 0, 0):
#                     # NEW OPERATOR (Blender 4.0+)
#                     bpy.ops.wm.obj_import(filepath=path)
#                 else:
#                     # OLD OPERATOR (Blender 3.6 and older)
#                     bpy.ops.import_scene.obj(filepath=path, use_split_objects=False)

#                 # Get the imported object
#                 # The importer selects the new objects, so we grab the selected one
#                 if not bpy.context.selected_objects:
#                     print(f"Warning: Nothing imported for {fname}")
#                     continue
                    
#                 obj = bpy.context.selected_objects[0]
                
#                 # Rename the mesh data so we can find it later
#                 obj.data.name = f"tpose_seq_{i}"
#                 loaded_meshes.append(obj.data)
                
#                 # Delete the temporary object immediately (we only keep the mesh data in memory)
#                 bpy.data.objects.remove(obj, do_unlink=True)
                
#             except Exception as e:
#                 print(f"Error loading {fname}: {e}")
#                 import traceback
#                 traceback.print_exc()
#                 continue
    
#     if not loaded_meshes:
#         print("No meshes were loaded successfully!")
#         return
    
#     # Create the Container Object
#     if SEQUENCE_OBJ_NAME in bpy.data.objects:
#         container = bpy.data.objects[SEQUENCE_OBJ_NAME]
#     else:
#         container = bpy.data.objects.new(SEQUENCE_OBJ_NAME, loaded_meshes[0])
#         bpy.context.collection.objects.link(container)
    
#     # Frame Handler to swap mesh data
#     def frame_handler(scene):
#         obj = bpy.data.objects.get(SEQUENCE_OBJ_NAME)
#         # Assuming frame 0 maps to index 0. 
#         # If your timeline starts at 1, use (scene.frame_current - 1)
#         idx = scene.frame_current
        
#         if obj and 0 <= idx < len(loaded_meshes):
#             obj.data = loaded_meshes[idx]

#     # Clear old handlers and append new one
#     bpy.app.handlers.frame_change_pre.clear()
#     bpy.app.handlers.frame_change_pre.append(frame_handler)
    
#     print(f"Sequence Loaded! {len(loaded_meshes)} frames ready.")

# if __name__ == "__main__":
#     load_tpose_sequence()