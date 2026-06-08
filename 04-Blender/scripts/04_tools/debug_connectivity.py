import bpy
import json
import gpu
from gpu_extras.batch import batch_for_shader
import os

# ================= CONFIGURATION =================
# JSON_PATH = "C:/Users/Eduardo/00-Local/Blender/MUSK/output/individual_muscle_to_skin_binding.json"
# JSON_PATH = "C:/Users/Eduardo/00-Local/Blender/MUSK/output/individual_muscle_to_skin_binding_FIX.json"
JSON_PATH = "C:/Users/Eduardo/00-Local/Blender/MUSK/output/individual_muscle_to_skin_binding_FIX_NEW.json"

SKIN_OBJ_NAME = "skin_layer-S1-ID"
SCENE_MUSCLE_SUFFIX = "_a-pose-ID" 

HANDLER_KEY = "muscle_binding_debug"

# [NEW] Filter Configuration
TARGET_MUSCLE = "r-vastus-lateralis" + SCENE_MUSCLE_SUFFIX
# =================================================

def draw_debug_lines():
    # 1. CLEANUP OLD LINES
    if HANDLER_KEY in bpy.app.driver_namespace:
        print("Removing old debug lines...")
        try:
            bpy.types.SpaceView3D.draw_handler_remove(bpy.app.driver_namespace[HANDLER_KEY], 'WINDOW')
        except ValueError:
            pass 
        del bpy.app.driver_namespace[HANDLER_KEY]

    # 2. LOAD DATA
    if not os.path.exists(JSON_PATH):
        print(f"Error: JSON file not found at {JSON_PATH}")
        return

    with open(JSON_PATH, 'r') as f:
        binding_data = json.load(f)

    skin_obj = bpy.data.objects.get(SKIN_OBJ_NAME)
    if not skin_obj:
        print(f"Error: Skin object '{SKIN_OBJ_NAME}' not found!")
        return
    
    mw_skin = skin_obj.matrix_world
    skin_mesh = skin_obj.data
    
    coords = []
    
    # 3. BUILD GEOMETRY
    stride = 1

    print(f"Visualizing binding for TARGET: {TARGET_MUSCLE}...")

    for m_name, vert_data in binding_data.items():
        
        # --- FILTER: Skip everything that is not the target muscle ---
        # We use 'in' to handle potential slight naming variations (e.g. suffixes)
        if m_name != TARGET_MUSCLE:
            continue
        # -------------------------------------------------------------

        # Find the muscle object in the scene
        m_obj = bpy.data.objects.get(m_name)
        print(m_obj)
        if not m_obj:
            # m_obj = bpy.data.objects.get(m_name + SCENE_MUSCLE_SUFFIX)
            m_obj = bpy.data.objects.get(m_name)

        if not m_obj:
            print(f"  [Warning] Muscle '{m_name}' not found in scene. Skipping.")
            continue

        print("HELLOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")   
        print(f"  Processing {m_name}...")
        mw_muscle = m_obj.matrix_world
        muscle_mesh = m_obj.data
        
        v_keys = list(vert_data.keys())[::stride]
        
        for i_str in v_keys:
            data = vert_data[i_str]
            i = int(i_str)
            
            if i >= len(muscle_mesh.vertices): continue
            
            # Start: Muscle
            start_pos = mw_muscle @ muscle_mesh.vertices[i].co
            
            # End: Skin
            tri_idx = data['tri_idx']
            bary = data['bary']
            
            poly = skin_mesh.polygons[tri_idx]
            p1 = mw_skin @ skin_mesh.vertices[poly.vertices[0]].co
            p2 = mw_skin @ skin_mesh.vertices[poly.vertices[1]].co
            p3 = mw_skin @ skin_mesh.vertices[poly.vertices[2]].co
            
            target_pos = (p1 * bary[0]) + (p2 * bary[1]) + (p3 * bary[2])
            
            coords.append(start_pos)
            coords.append(target_pos)

    # 4. SETUP DRAWING
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'LINES', {"pos": coords})
    
    def draw():
        shader.bind()
        shader.uniform_float("color", (1, 0.8, 0, 1)) 
        batch.draw(shader)
        
    handler = bpy.types.SpaceView3D.draw_handler_add(draw, (), 'WINDOW', 'POST_VIEW')
    bpy.app.driver_namespace[HANDLER_KEY] = handler
    
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()
            
    print(f"Drawing {len(coords)//2} lines for {TARGET_MUSCLE}.")

if __name__ == "__main__":
    draw_debug_lines()