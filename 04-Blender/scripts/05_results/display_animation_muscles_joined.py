import bpy
import os
import re

# --- CONFIGURATION ---
# List of ROOT directories containing the "frame_XXXX" folders
DIRECTORIES_TO_LOAD = [
    # {
    #     "path": r"T:\static00\S1\exported_ind_musc_meshes_0_300_shot_001\S1\shot_001_captury",
    #     "name": "exported_ind_musc_meshes_0_300_shot_001", # The name of the resulting object
    #     "frame_prefix": "frame_"     # The folder name pattern (e.g., "frame_0001")
    # },
    {
        "path": r"T:\static00\S1\exported_ind_musc_meshes_1000_1300_shot_001\S1\shot_001_captury",
        "name": "exported_ind_musc_meshes_1000_1300_shot_001",       # Name of the object in Blender
        "frame_prefix": "frame_"     # The folder name pattern (e.g., "frame_0001")
    },
    {
        "path": r"T:\static00\S1\exported_ind_musc_meshes_1700_2000_shot_001\S1\shot_001_captury",
        "name": "exported_ind_musc_meshes_1700_2000_shot_001",
        "frame_prefix": "frame_"     # The folder name pattern (e.g., "frame_0001")
    },
    {
        "path": r"T:\static00\S1\exported_ind_musc_meshes_50_350_shot_002\S1\shot_002_captury",
        "name": "exported_ind_musc_meshes_50_350_shot_002",
        "frame_prefix": "frame_"
    },
    {
        "path": r"T:\static00\S1\exported_ind_musc_meshes_100_400_shot_003\S1\shot_003_captury",
        "name": "exported_ind_musc_meshes_100_400_shot_003",
        "frame_prefix": "frame_"
    },
    {
        "path": r"T:\static00\S1\exported_ind_musc_meshes_40_340_shot_004\S1\shot_004_captury",
        "name": "exported_ind_musc_meshes_40_340_shot_004",
        "frame_prefix": "frame_"
    }
]

FILE_EXTENSION = ".obj"
# ---------------------

# Storage: { "ObjectName": { 1: mesh_data_1, 2: mesh_data_2 } }
if "GLOBAL_MUSCLE_MAP" not in globals():
    GLOBAL_MUSCLE_MAP = {}

def extract_frame_number(foldername):
    # "frame_0001" -> 1
    nums = re.findall(r'\d+', foldername)
    if nums:
        return int(nums[-1])
    return None

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def load_muscle_sequences():
    global GLOBAL_MUSCLE_MAP
    GLOBAL_MUSCLE_MAP.clear()

    window = bpy.context.window_manager.windows[0]
    with bpy.context.temp_override(window=window):
        
        for entry in DIRECTORIES_TO_LOAD:
            root_path = entry["path"]
            obj_name = entry["name"]
            prefix = entry["frame_prefix"]
            
            if not os.path.exists(root_path):
                print(f"[SKIP] Path not found: {root_path}")
                continue

            # 1. Find all "frame_XXXX" folders
            all_items = os.listdir(root_path)
            frame_folders = [d for d in all_items 
                             if os.path.isdir(os.path.join(root_path, d)) and d.startswith(prefix)]
            
            if not frame_folders:
                print(f"[SKIP] No frame folders found in {root_path}")
                continue

            print(f"--- Processing {obj_name}: Found {len(frame_folders)} frame folders ---")
            
            frame_dict = {} # { 1: mesh_data, 2: mesh_data ... }
            first_mesh = None
            
            # Process each frame folder
            for folder_name in frame_folders:
                frame_num = extract_frame_number(folder_name)
                if frame_num is None: continue
                
                folder_full_path = os.path.join(root_path, folder_name)
                
                # A. Find all OBJs inside this frame folder
                muscle_files = [f for f in os.listdir(folder_full_path) if f.endswith(FILE_EXTENSION)]
                
                # CRITICAL: Sort them so they are joined in the EXACT same order every time
                # Otherwise topology changes and rigging fails.
                muscle_files.sort(key=natural_sort_key)
                
                if not muscle_files: continue
                
                imported_objects = []
                
                try:
                    # B. Import all muscles for this frame
                    for m_file in muscle_files:
                        m_path = os.path.join(folder_full_path, m_file)
                        
                        if bpy.app.version >= (4, 0, 0):
                            bpy.ops.wm.obj_import(filepath=m_path)
                        else:
                            bpy.ops.import_scene.obj(filepath=m_path, use_split_objects=False)
                        
                        # Gather imported objects
                        # The importer selects new objects, so we grab selected
                        for obj in bpy.context.selected_objects:
                            if obj.type == 'MESH':
                                imported_objects.append(obj)
                    
                    if not imported_objects: continue
                    
                    # C. Join them into one
                    # Deselect everything first
                    bpy.ops.object.select_all(action='DESELECT')
                    
                    # Select all imported muscles
                    for obj in imported_objects:
                        obj.select_set(True)
                    
                    # Set active object (needed for join)
                    bpy.context.view_layer.objects.active = imported_objects[0]
                    
                    # Join
                    bpy.ops.object.join()
                    
                    # The result is the active object
                    joined_obj = bpy.context.view_layer.objects.active
                    joined_obj.data.name = f"{obj_name}_{frame_num}_joined"
                    
                    # Store mesh data
                    frame_dict[frame_num] = joined_obj.data
                    
                    if first_mesh is None: first_mesh = joined_obj.data
                    
                    # Delete the joined object container (we keep the mesh data in memory)
                    bpy.data.objects.remove(joined_obj, do_unlink=True)
                    
                    print(f"Frame {frame_num}: Joined {len(muscle_files)} muscles.")
                    
                except Exception as e:
                    print(f"Error processing frame {frame_num}: {e}")
                    # Cleanup if failed
                    for obj in imported_objects:
                        try: bpy.data.objects.remove(obj)
                        except: pass
                    continue

            if not frame_dict: continue

            # Store in global cache
            GLOBAL_MUSCLE_MAP[obj_name] = frame_dict

            # Create Container Object in Scene
            if obj_name in bpy.data.objects:
                container = bpy.data.objects[obj_name]
                container.data = first_mesh
            else:
                container = bpy.data.objects.new(obj_name, first_mesh)
                bpy.context.collection.objects.link(container)
            
            print(f"[SUCCESS] {obj_name} ready. Frames {min(frame_dict.keys())}-{max(frame_dict.keys())}")

    # --- DEFINE HANDLER ---
    def joined_muscle_sync_handler(scene):
        current_frame = scene.frame_current
        
        for obj_name, frame_map in GLOBAL_MUSCLE_MAP.items():
            obj = bpy.data.objects.get(obj_name)
            
            if obj:
                # Exact frame match
                if current_frame in frame_map:
                    if obj.data != frame_map[current_frame]:
                        obj.data = frame_map[current_frame]
                else:
                    # Optional: Hold last known frame or hide?
                    pass

    # Register
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_pre.append(joined_muscle_sync_handler)
    print("Muscle Join Handler Registered!")

if __name__ == "__main__":
    load_muscle_sequences()