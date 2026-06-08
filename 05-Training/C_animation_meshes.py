import bpy
import os

# --- CONFIGURATION ---

SHOT = "shot_001"

# OBJ_DIR = rf"C:\Users\Eduardo\00-Local\Blender\SOMA\exports_multiple\S1\{SHOT}"

SUB = "S4"
OBJ_DIR = rf"C:\Users\Eduardo\00-Local\Blender\SOMA\exports_multiple\{SUB}\validation\validation_600_900"


# PREFIX = "skin_frame_"              
# PREFIX = "muscle_frame_"                    
PREFIX = "indiv_muscles_unified_frame_"                    

TARGET_OBJ_NAME = bpy.context.active_object.name

# ---------------------

# Global dictionary to cache the vertex coordinates in RAM
if "soma_vertex_cache" not in locals():
    soma_vertex_cache = {}

def load_obj_vertices(filepath):
    """Fast, lightweight parser to extract only vertex coordinates."""
    verts = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                # Grab the X, Y, Z float values
                verts.extend([float(x) for x in line.strip().split()[1:4]])
    return verts

def build_cache():
    """Reads all OBJs in the directory and stores them in RAM."""
    global soma_vertex_cache
    soma_vertex_cache.clear()
    
    if not os.path.exists(OBJ_DIR):
        print(f"[ERROR] Directory not found: {OBJ_DIR}")
        return

    # Find and sort all matching OBJ files
    files = [f for f in os.listdir(OBJ_DIR) if f.startswith(PREFIX) and f.endswith(".obj")]
    
    print(f"Caching {len(files)} frames into RAM. Please wait...")
    
    for f in files:
        # Extract the frame number (e.g., "skin_frame_0001.obj" -> 1)
        frame_str = f.replace(PREFIX, "").replace(".obj", "")
        try:
            frame_idx = int(frame_str)
        except ValueError:
            continue # Skip files that don't match the numbering format
            
        filepath = os.path.join(OBJ_DIR, f)
        soma_vertex_cache[frame_idx] = load_obj_vertices(filepath)
        
    print(f"[SUCCESS] Caching complete! {len(soma_vertex_cache)} frames ready.")

def frame_change_handler(scene):
    """Fires every time the timeline frame changes."""
    global soma_vertex_cache
    frame = scene.frame_current
    
    # If we have cached data for this frame
    if frame in soma_vertex_cache:
        obj = bpy.data.objects.get(TARGET_OBJ_NAME)
        if obj and obj.type == 'MESH':
            # Rapidly overwrite the base mesh vertices
            try:
                obj.data.vertices.foreach_set('co', soma_vertex_cache[frame])
                obj.data.update() # Tell Blender to redraw
            except Exception as e:
                print(f"Topology mismatch on frame {frame}: {e}")

# --- EXECUTION ---
# 1. Clear any existing handlers (prevents duplicates if you run the script twice)
bpy.app.handlers.frame_change_pre.clear()

# 2. Build the RAM cache
build_cache()

# 3. Attach the handler to the timeline
bpy.app.handlers.frame_change_pre.append(frame_change_handler)

print(f"--- Ready! Press Play in the timeline to animate '{TARGET_OBJ_NAME}'. ---")