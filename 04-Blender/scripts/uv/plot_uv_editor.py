"""
Use Blender to print UV points that failed to be detected.
"""

import json
import bpy
import json
from mathutils import Vector

# Name for the new object that will visualize the points
VIS_OBJECT_NAME = "Failed_UV_Points_Vis"

tracking_folder = "C:/Users/ealvarad/00 - Local/02 - Python/pythonProjectLocal/blender/inside-humans/uv_detections_charuco-suit/missed.json"

print(f"Starting UV point visualization script...")

with open(tracking_folder, "r") as f:
    failed_uv_data = json.load(f)

# --- 2. Prepare Data for Mesh Creation ---
# We need lists of 3D vertex coordinates and corresponding UV coordinates
vertex_coords_local = []  # We'll store 3D coordinates here
uv_coords_list = []  # We'll store the corresponding (u,v) coordinates here
feature_ids = []  # Store the original ID for potential reference

for feature_id_str, uv_list in failed_uv_data.items():
    try:
        u = float(uv_list[0])
        v = float(uv_list[1])

        # Add the UV coordinate to our list
        uv_coords_list.append(Vector((u, v)))

        # Add a corresponding 3D vertex. Placing it at (u, v, 0) in local space
        # helps visualize in 3D view too, though (0,0,0) would also work for UV view.
        vertex_coords_local.append(Vector((u, v, 0.0)))

        feature_ids.append(feature_id_str)  # Keep track of which point is which

    except (ValueError, TypeError, IndexError) as e:
        print(f"Warning: Skipping invalid data entry for '{feature_id_str}'. Data: {uv_list}. Error: {e}")

if not vertex_coords_local:
    raise Exception("ERROR: No valid points prepared after processing JSON. Check data format.")

print(f"Prepared {len(vertex_coords_local)} points for mesh creation.")

# --- 3. Create New Mesh Object ---
# Check if visualization object already exists, remove if necessary
if VIS_OBJECT_NAME in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects[VIS_OBJECT_NAME], do_unlink=True)
if VIS_OBJECT_NAME in bpy.data.meshes:
    bpy.data.meshes.remove(bpy.data.meshes[VIS_OBJECT_NAME])

mesh_data = bpy.data.meshes.new(name=VIS_OBJECT_NAME)
vis_object = bpy.data.objects.new(VIS_OBJECT_NAME, mesh_data)

# --- 4. Populate Mesh with Vertices (No Faces/Edges Needed for UV View) ---
num_verts = len(vertex_coords_local)
mesh_data.vertices.add(num_verts)

# Flatten the list of Vectors for foreach_set
flat_vertex_coords = [coord for vec in vertex_coords_local for coord in vec]
mesh_data.vertices.foreach_set("co", flat_vertex_coords)

# --- 5. Create UV Layer and Assign UVs ---
# UVs are stored per loop, not per vertex directly.
# Workaround: Create dummy loops/faces just to store the UVs.
# Simplest dummy: one loop per vertex (requires faces).
# Let's create one face per vertex (a degenerate point-face). Blender might handle this.
# *Correction*: Blender needs faces with >= 3 vertices.
# Alternative: Create one large face using all vertices? No, loops won't match.
# Let's use the BMesh approach for cleaner handling, creating tiny triangles.

import bmesh

bm = bmesh.new()

# Ensure we have a UV layer available
uv_layer = bm.loops.layers.uv.new("FailedUVs")  # Use verify() if it might exist

epsilon = 1e-5  # Tiny offset for degenerate triangles, adjust if needed

vert_list_mapping = []  # Store mapping if needed: index -> feature_id

for i, target_uv in enumerate(uv_coords_list):
    # Use the 3D coords we prepared earlier, or just place near origin
    # Using (u,v,0) might spread them out in 3D view too
    # base_3d_coord = vertex_coords_local[i]
    # Let's place all base vertices at origin for simplicity in 3D view
    base_3d_coord = Vector((0.0, 0.0, 0.0))

    # Create 3 vertices for a tiny triangle (doesn't really matter in UV view)
    # The first vertex's loop will hold our target UV
    v1 = bm.verts.new(base_3d_coord)
    v2 = bm.verts.new(base_3d_coord + Vector((epsilon, 0, 0)))  # Tiny offset
    v3 = bm.verts.new(base_3d_coord + Vector((0, epsilon, 0)))  # Tiny offset

    # Create a face from these vertices
    try:
        face = bm.faces.new((v1, v2, v3))
        face.normal_update()  # Update normal just in case

        # Assign the target UV coordinate to the loop corresponding to the first vertex (v1)
        # The UVs for v2, v3 loops don't really matter for visualization purposes here.
        face.loops[0][uv_layer].uv = target_uv
        # Optionally set other loop UVs slightly offset if needed for interpolation checks later
        # face.loops[1][uv_layer].uv = target_uv + Vector((epsilon, 0))
        # face.loops[2][uv_layer].uv = target_uv + Vector((0, epsilon))

        # face.loops[0][uv_layer].uv = target_uv
        face.loops[1][uv_layer].uv = target_uv
        face.loops[2][uv_layer].uv = target_uv

        # Store the feature ID associated with v1 (the vertex at the target UV)
        # Note: BMesh vertex indices change after writing to mesh.
        # We might need a way to map back if needed later. For now, just visualize.

    except ValueError as e:
        # This can happen if vertices are numerically identical creating degenerate face
        print(f"Warning: Could not create face for UV {target_uv}. Vertices might be too close. Error: {e}")
        # Clean up verts if face fails? Or let bmesh handle it.
        bm.verts.remove(v1)
        bm.verts.remove(v2)
        bm.verts.remove(v3)

# Write the BMesh data back to the mesh object
bm.to_mesh(mesh_data)
bm.free()  # Release the BMesh instance

mesh_data.update()  # Update mesh data display

# --- 6. Link Object to Scene ---
bpy.context.collection.objects.link(vis_object)

print(f"\n--- Script Finished ---")
print(f"Created object '{VIS_OBJECT_NAME}' containing visualization points.")
print(f"To view: Select the '{VIS_OBJECT_NAME}' object, then go to the 'UV Editing' workspace.")
print(f"The points should appear overlaid on the UV grid at their corresponding (u,v) coordinates.")
print(f"Make sure you are viewing the 'FailedUVs' UV Map for the '{VIS_OBJECT_NAME}' object.")
