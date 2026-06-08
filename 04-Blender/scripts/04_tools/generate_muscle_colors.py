import bpy
import random

def assign_random_colors_full(col_name):
    # 1. Get the collection
    if col_name not in bpy.data.collections:
        print(f"Error: Collection '{col_name}' not found.")
        return
    
    collection = bpy.data.collections[col_name]
    
    # 2. Iterate through objects
    for obj in collection.objects:
        if obj.type == 'MESH':
            
            # Create a new material
            mat = bpy.data.materials.new(name=f"Random_Mat_{obj.name}")
            mat.use_nodes = True
            
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            
            if bsdf:
                # Generate random color (R, G, B, Alpha)
                r = random.random()
                g = random.random()
                b = random.random()
                alpha = 1.0

                r = 1
                g = 0.2
                b = 0

                color_values = (r, g, b, alpha)
                
                # --- APPLY COLORS ---
                
                # 1. Rendered Color (Principled BSDF)
                bsdf.inputs['Base Color'].default_value = color_values
                
                # 2. Material Viewport Color (Solid Mode)
                # Found in: Material Properties > Viewport Display > Color
                mat.diffuse_color = color_values
                
                # 3. Object Wireframe Color
                # Found in: Object Properties > Viewport Display > Color
                obj.color = color_values
                
                # --- ASSIGN MATERIAL ---
                obj.data.materials.clear()
                obj.data.materials.append(mat)

# --- CONFIGURATION ---
collection_name = "FBX-S1-Complex" 
collection_name = "Static_Canonical_Marker_Spheres_TPose" 

assign_random_colors_full(collection_name)