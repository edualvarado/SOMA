import bpy
import json
import mathutils
import os

# ================= CONFIGURATION =================

SKIN_OBJ_NAME = "skin_layer-S1-ID"
MUSCLE_COLLECTION_NAME = "FBX-S1-Complex-Muscles-Only-APose"
# BONE_NAME_REPLACE = "_a-pose.nonworking"

# 3. Where to save the map
OUTPUT_PATH = "C:/Users/Eduardo/00-Local/Blender/MUSK/output/individual_muscle_to_skin_binding_FIX_NEW.json"

# 4. Maximum reach (meters). If skin is further than this, ignore the vertex.
MAX_DISTANCE = 0.03
# =================================================

def get_barycentric_coords(p, a, b, c):
    """Calculates weights (u, v, w) for point p inside triangle abc"""
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = v0.dot(v0), v0.dot(v1), v1.dot(v1)
    d20, d21 = v2.dot(v0), v2.dot(v1)
    denom = d00 * d11 - d01 * d01
    
    # Degenerate triangle check
    if abs(denom) < 1e-8: return 0.33, 0.33, 0.33
    
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w

def main():
    print("-" * 50)
    print("STARTING MUSCLE -> SKIN BINDING")
    
    # 1. Get Objects
    skin_obj = bpy.data.objects.get(SKIN_OBJ_NAME)
    muscle_coll = bpy.data.collections.get(MUSCLE_COLLECTION_NAME)
    
    if not skin_obj or not muscle_coll:
        print(f"ERROR: Skin object '{SKIN_OBJ_NAME}' or Muscle collection '{MUSCLE_COLLECTION_NAME}' not found.")
        return

    # Get the evaluated dependency graph ONCE
    depsgraph = bpy.context.evaluated_depsgraph_get()
    
    # Get evaluated skin object and its world matrix and mesh data ONCE
    evaluated_skin_obj = skin_obj.evaluated_get(depsgraph)
    mw_skin = evaluated_skin_obj.matrix_world
    skin_mesh_data = evaluated_skin_obj.data # This is the mesh data after modifiers
    
    print(f"Skin object '{skin_obj.name}' evaluated world matrix:\n{mw_skin}")
    if skin_mesh_data.vertices:
        print(f"  First skin vertex local coord: {skin_mesh_data.vertices[0].co}")
        print(f"  First skin vertex world coord: {mw_skin @ skin_mesh_data.vertices[0].co}")


    # Gather muscles and sort them alphabetically (crucial for consistency)
    muscles = [o for o in muscle_coll.objects if o.type == 'MESH']
    muscles.sort(key=lambda o: o.name)
    
    binding_data = {}
    
    # 2. Iterate Muscles
    for m_obj in muscles:
        print(f"Binding {m_obj.name}...")
        
        # Get evaluated muscle object and its world matrix and mesh data for EACH muscle
        evaluated_m_obj = m_obj.evaluated_get(depsgraph)
        muscle_mesh_data = evaluated_m_obj.data # This is the mesh data after modifiers
        mw_muscle = evaluated_m_obj.matrix_world
        
        print(f"  Muscle '{m_obj.name}' evaluated world matrix:\n{mw_muscle}")
        if muscle_mesh_data.vertices:
            print(f"    First muscle vertex local coord: {muscle_mesh_data.vertices[0].co}")
            print(f"    First muscle vertex world coord: {mw_muscle @ muscle_mesh_data.vertices[0].co}")
        
        # This dict stores: Vertex_Index -> {Triangle, Barycentric}
        m_data = {}
        
        for i, v in enumerate(muscle_mesh_data.vertices): # Use evaluated muscle_mesh_data
            # Calculate Muscle Vertex Global Position
            world_pos = mw_muscle @ v.co # Use evaluated mw_muscle
            
            # Find Closest Point on Skin
            # returns: (success, location, normal, index)
            # Use evaluated_skin_obj for closest_point_on_mesh
            success, loc, norm, tri_idx = evaluated_skin_obj.closest_point_on_mesh(world_pos, distance=MAX_DISTANCE)

            if success:
                # We hit the skin! Calculate Barycentric weights.
                poly = skin_mesh_data.polygons[tri_idx] # Use evaluated skin_mesh_data
                
                # Get world positions of the skin triangle corners
                p1 = mw_skin @ skin_mesh_data.vertices[poly.vertices[0]].co # Use evaluated mw_skin
                p2 = mw_skin @ skin_mesh_data.vertices[poly.vertices[1]].co # Use evaluated mw_skin
                p3 = mw_skin @ skin_mesh_data.vertices[poly.vertices[2]].co # Use evaluated mw_skin
                
                u, v_coord, w = get_barycentric_coords(loc, p1, p2, p3)
                
                # Save Data
                m_data[str(i)] = {
                    "tri_idx": tri_idx,
                    "bary": [u, v_coord, w],
                    "skin_vert_indices": list(poly.vertices) # Helpful for debugging
                }
        
        # Clean name: remove suffixes like "_a-pose.001" if you want clean keys
        # Ensure BONE_NAME_REPLACE matches the suffix of your muscle objects
        # clean_name = m_obj.name.replace(BONE_NAME_REPLACE, "").split(".")[0]
        # binding_data[clean_name] = m_data
        binding_data[m_obj.name] = m_data

    # 3. Save to JSON
    folder = os.path.dirname(OUTPUT_PATH)
    if not os.path.exists(folder): os.makedirs(folder)
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(binding_data, f, indent=2) # Added indent for readability
        
    print(f"DONE. Saved binding map to: {OUTPUT_PATH}")
    print("-" * 50)

if __name__ == "__main__":
    main()