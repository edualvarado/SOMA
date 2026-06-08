import trimesh
import numpy as np
import os
import glob
import json
from scipy.spatial import KDTree
from pathlib import Path

subject = "S1"

# --- CONFIGURATION ---
# Use Linux absolute paths since we're running on Linux server
# SKIN_PATH = Path("/CT/MUSK/static00/musk_training_data_21_11_24/layers/apose/skin_layer_apose_baked.obj")
# MUSCLE_DIR = Path("/CT/MUSK/static00/musk_training_data_21_11_24/layers/apose/muscle_meshes_apose")
# OUTPUT_JSON_PATH = Path("/CT/MUSK/static00/musk_training_data_21_11_24/layers/apose/skin_vertex_ids.json")

SKIN_PATH = Path(f"/CT/SOMA/static00/{subject}/layers/apose/skin_layer-{subject}-APose_baked.obj")
MUSCLE_DIR = Path(f"/CT/SOMA/static00/{subject}/layers/apose/muscle_meshes_apose")
OUTPUT_JSON_PATH = Path("/CT/SOMA/static00/musk_training_data_21_11_24/layers/apose/skin_vertex_ids.json")

print(f"SKIN_PATH: {SKIN_PATH}")
print(f"MUSCLE_DIR: {MUSCLE_DIR}")
print(f"SKIN_PATH exists: {SKIN_PATH.exists()}")
print(f"MUSCLE_DIR exists: {MUSCLE_DIR.exists()}")
# ---------------------

def load_obj_simple(file_path):
    """
    Directly reads 'v' lines to guarantee vertex count matches Blender.
    """
    vertices = []
    # We don't even need faces for the ID map, just vertices
    print(f"Parsing raw file: {file_path}")
    
    with open(file_path, 'r') as f:
        for line in f:
            # Check for 'v' followed by space to avoid 'vt', 'vn', etc.
            if line.startswith('v '):
                # Parse x, y, z
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    return np.array(vertices, dtype=np.float32)

def main():
    print("-" * 60)
    print("STARTING ROBUST ID GENERATION")
    
    # 1. Load Skin using YOUR robust method
    skin_verts = load_obj_simple(SKIN_PATH)
    print(f"Skin Vertices Loaded: {len(skin_verts)}")
    
    # Double check against expected count
    if len(skin_verts) == 23752:
        print("SUCCESS: Exact match with Blender count.")
    else:
        print(f"WARNING: Count is {len(skin_verts)}. If Blender says 23752, check you exported the right object!")

    # 2. Load Muscles & Build ID Map
    # Sort files to match Blender's os.listdir / alphabetical order
    muscle_files = sorted(glob.glob(os.path.join(MUSCLE_DIR, "*.obj")))
    print(f"Found {len(muscle_files)} muscles.")
    
    all_verts = []
    all_ids = []
    
    print("Loading muscles (using standard loader)...")
    for m_id, f_path in enumerate(muscle_files):
        # Trimesh is fine here because we just need the cloud of points for the KDTree
        # We don't care if it splits vertices for muscles
        m = trimesh.load(f_path, process=False) 
        all_verts.append(m.vertices)
        
        # Assign ID
        ids = np.full(len(m.vertices), m_id, dtype=np.int32)
        all_ids.append(ids)

    # 3. Build Spatial Index (KDTree)
    print("Building KDTree...")
    stacked_verts = np.vstack(all_verts)
    stacked_ids = np.concatenate(all_ids)
    tree = KDTree(stacked_verts)
    
    # 4. Query Nearest Muscle
    print("Querying nearest muscles...")
    dists, indices = tree.query(skin_verts, k=1)
    skin_ids = stacked_ids[indices]
    
    # 5. Save
    print(f"Saving {len(skin_ids)} IDs to {OUTPUT_JSON_PATH}")
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(skin_ids.tolist(), f)
        
    print("Done.")
    print("-" * 60)

if __name__ == "__main__":
    main()