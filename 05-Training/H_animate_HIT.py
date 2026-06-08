import bpy
import os
from pathlib import Path

# --- CONFIGURATION ---
SOURCE_BASE = Path(r"C:\Users\Eduardo\00-Local\Blender\SOMA\smpl\muscles")
FILE_NAME = "LT_mesh.obj"

# ALIGNMENT SETTINGS
GROUND_ANIMATION = True  
UP_AXIS = 1              

# NEW: FRAME LIMIT
MAX_FRAME = 300
# ---------------------

def get_obj_sequence():
    sequence = []
    base_path = str(SOURCE_BASE)
    
    print(f"Scanning directory up to frame {MAX_FRAME}: {base_path} ...")
    
    try:
        entries = os.scandir(base_path)
    except FileNotFoundError:
        print(f"[ERROR] Could not find directory.")
        return sequence

    for entry in entries:
        if entry.is_dir() and entry.name.startswith("frame_"):
            try:
                frame_num = int(entry.name.split('_')[1])
            except ValueError:
                continue
                
            # --- NEW: Skip folders past our maximum frame ---
            if frame_num > MAX_FRAME:
                continue
                
            target_file = os.path.join(entry.path, FILE_NAME)
            
            if os.path.exists(target_file):
                sequence.append((frame_num, target_file))
                
    sequence.sort(key=lambda x: x[0])
    return sequence

def get_vertex_count(filepath):
    """Quickly counts vertices without fully parsing."""
    count = 0
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                count += 1
    return count

def create_base_mesh_from_obj(filepath, obj_name="Muscles_Animated"):
    verts = []
    faces = []
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.split()[1:]
                face = []
                for p in parts:
                    v_idx = int(p.split('/')[0]) - 1
                    face.append(v_idx)
                faces.append(face)
    
    # Ground the Base Mesh
    if GROUND_ANIMATION and verts:
        min_val = min(v[UP_AXIS] for v in verts)
        for v in verts:
            v[UP_AXIS] -= min_val
            
    mesh = bpy.data.meshes.new(obj_name + "_Mesh")
    mesh.from_pydata([tuple(v) for v in verts], [], faces)
    mesh.update()
    
    obj = bpy.data.objects.new(obj_name, mesh)
    bpy.context.collection.objects.link(obj)
    
    for polygon in mesh.polygons:
        polygon.use_smooth = True
        
    return obj

def read_vertex_coords(filepath):
    coords = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                coords.extend((float(parts[1]), float(parts[2]), float(parts[3])))
    return coords

def main():
    sequence = get_obj_sequence()
    if not sequence:
        print("[ERROR] No OBJ files found.")
        return

    print("Analyzing topology stability...")
    
    # 1. Figure out the "correct" topology by looking at the last frame in our truncated list
    last_frame_file = sequence[-1][1]
    target_v_count = get_vertex_count(last_frame_file)
    target_floats = target_v_count * 3
    print(f"Target stable topology is {target_v_count} vertices.")

    # 2. Find the FIRST frame that matches this topology to use as our Base
    base_index = 0
    for i, (f_num, f_path) in enumerate(sequence):
        if get_vertex_count(f_path) == target_v_count:
            base_index = i
            break
        else:
            print(f" -> Skipping warm-up frame {f_num} (Mismatching vertex count)")

    # 3. Build the Base Mesh
    base_frame_num, base_file = sequence[base_index]
    print(f"Building Base Mesh from frame {base_frame_num}...")
    
    base_obj = create_base_mesh_from_obj(base_file, "Muscles_Animated")
    
    bpy.context.view_layer.objects.active = base_obj
    base_obj.select_set(True)
    
    sk_basis = base_obj.shape_key_add(name="Basis")
    base_obj.data.shape_keys.use_relative = True

    bpy.context.scene.frame_start = sequence[0][0]
    bpy.context.scene.frame_end = sequence[-1][0]

    # 4. Iterate and bake Shape Keys for the REST of the sequence
    for frame_num, obj_file in sequence[base_index + 1:]:
        
        flat_coords = read_vertex_coords(obj_file)
        
        # Double check to ensure we only process perfectly matching frames
        if len(flat_coords) != target_floats:
            print(f"[WARNING] Topology mismatch at frame {frame_num}. Skipping.")
            continue
            
        # Ground the Shape Key Vertices
        if GROUND_ANIMATION:
            min_val = min(flat_coords[UP_AXIS::3])
            for i in range(UP_AXIS, len(flat_coords), 3):
                flat_coords[i] -= min_val
                
        sk = base_obj.shape_key_add(name=f"Frame_{frame_num}")
        sk.data.foreach_set("co", flat_coords)
        
        sk.value = 0.0
        sk.keyframe_insert(data_path="value", frame=frame_num - 1)
        
        sk.value = 1.0
        sk.keyframe_insert(data_path="value", frame=frame_num)
        
        sk.value = 0.0
        sk.keyframe_insert(data_path="value", frame=frame_num + 1)

    print(f"[SUCCESS] Built grounded Muscle animation up to frame {MAX_FRAME}!")
    print("Press Spacebar to play.")

main()