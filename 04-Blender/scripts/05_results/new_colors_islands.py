import bpy
import random
import colorsys

def create_random_material(index, total):
    """Generates a distinct color material based on HSV logic."""
    mat_name = f"Muscle_Mat_{index}"
    
    # Check if material exists
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        
        # Generate Color (Cycle through Hue)
        # We use Golden Ratio (0.618) to ensure colors are distinct and not similar neighbors
        hue = (index * 0.618033988749895) % 1.0 
        saturation = 0.8 + (random.random() * 0.2) # High saturation
        value = 0.8 + (random.random() * 0.2)      # High brightness
        
        # Convert to RGB
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # Set BSDF Color
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs['Base Color'].default_value = (r, g, b, 1.0)
            # Make it slightly shiny/fleshy
            bsdf.inputs['Roughness'].default_value = 0.4
            bsdf.inputs['Subsurface Weight'].default_value = 0.1
            bsdf.inputs['Subsurface Radius'].default_value = (1.0, 0.2, 0.1) # Reddish Scatter

    return mat

def color_islands():
    obj = bpy.context.active_object
    
    if not obj or obj.type != 'MESH':
        print("Please select a Mesh object.")
        return

    print(f"Processing '{obj.name}'...")
    original_name = obj.name
    
    # 1. Clear existing materials
    obj.data.materials.clear()
    
    # 2. Separate by Loose Parts (Islands)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.separate(type='LOOSE')
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # 3. Process the separated parts
    # 'selected_objects' will contain all the muscle parts now
    muscles = bpy.context.selected_objects
    
    print(f"Found {len(muscles)} individual muscles/islands.")
    
    for i, muscle in enumerate(muscles):
        # Create/Get unique material
        mat = create_random_material(i, len(muscles))
        
        # Assign to object
        if muscle.data.materials:
            muscle.data.materials[0] = mat
        else:
            muscle.data.materials.append(mat)
    
    # 4. Join them back together
    # Ensure all are selected
    for muscle in muscles:
        muscle.select_set(True)
    
    # Make one active (doesn't matter which, but ensures Join works)
    bpy.context.view_layer.objects.active = muscles[0]
    
    bpy.ops.object.join()
    
    # 5. Restore Name and Cleanup
    final_obj = bpy.context.active_object
    final_obj.name = original_name
    
    print(f"[SUCCESS] Colored {len(muscles)} muscles and rejoined into '{final_obj.name}'.")

if __name__ == "__main__":
    color_islands()