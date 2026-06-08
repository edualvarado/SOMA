import trimesh
import glob
import os

# --- Configuration ---
# folder_path = "path/to/individual_muscles"
folder_path = "../../../../../static00/musk_training_data_21_11_24/layers/tpose/muscle_meshes_tpose"

# ---------------------

obj_files = glob.glob(os.path.join(folder_path, "*.obj"))
total_verts = 0
total_faces = 0

print(f"Found {len(obj_files)} meshes in {folder_path}...")

for f in obj_files:
    # process=False loads faster and doesn't alter the mesh
    mesh = trimesh.load(f, process=False) 
    total_verts += len(mesh.vertices)
    total_faces += len(mesh.faces)

print("-" * 30)
print(f"Total Vertices: {total_verts:,}")
print(f"Total Faces:    {total_faces:,}")
print("-" * 30)