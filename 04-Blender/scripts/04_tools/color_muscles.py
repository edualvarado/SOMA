import bpy
import random

# 1. Select all your RAW muscle objects first!
selected_objects = bpy.context.selected_objects

# Sort them by name so IDs are consistent with your Python training script
selected_objects.sort(key=lambda x: x.name)

print(f"Processing {len(selected_objects)} muscles...")

for i, obj in enumerate(selected_objects):
    # Set the active object
    bpy.context.view_layer.objects.active = obj
    
    # Create a Vertex Color layer if it doesn't exist
    if not obj.data.vertex_colors:
        obj.data.vertex_colors.new()
    
    color_layer = obj.data.vertex_colors.active
    
    # Use 1/255 increments. Much safer for OBJ export precision.
    # ID 1 = 1/255, ID 2 = 2/255...
    val = (i + 1) / 255.0
    
    unique_color = (val, 0.0, 0.0, 1.0)
    
    # Assign to all loops (corners of faces)
    for data in color_layer.data:
        data.color = unique_color
        
    print(f"Assigned ID {i+1} (Color: {val:.4f}) to {obj.name}")
    
# Join them into one "Source" object for easier transfer
bpy.ops.object.join()
bpy.context.active_object.name = "Raw_Muscles_Combined"