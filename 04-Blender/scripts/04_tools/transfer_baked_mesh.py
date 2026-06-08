import os
import shutil
import sys
from pathlib import Path

def transfer_vertex_colors_raw(source_obj_path, target_obj_path, output_obj_path):
    """
    Reads RGB values from Source OBJ vertex lines and appends them to Target OBJ.
    Also handles MTL renaming/copying.
    """
    source_path = Path(source_obj_path)
    target_path = Path(target_obj_path)
    output_path = Path(output_obj_path)
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Extracting Colors from Source: {source_path.name}")
    colors = []
    
    # 1. READ SOURCE COLORS
    with open(source_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                # Format: v x y z r g b
                # We check if we have enough parts for color (at least 7 items: v + 3 coords + 3 colors)
                if len(parts) >= 7:
                    # Extract r, g, b (indices 4, 5, 6)
                    colors.append(parts[4:7]) 
                else:
                    # Fallback to white or black if source is missing color on a line
                    colors.append(["0.0", "0.0", "0.0"]) 
    
    print(f"      -> Extracted {len(colors)} color entries.")

    # 2. HANDLE MTL FILE
    print(f"[2/4] Handling Material File...")
    source_mtl = source_path.with_suffix('.mtl')
    output_mtl_name = output_path.with_suffix('.mtl').name
    output_mtl_path = output_path.with_suffix('.mtl')

    # Copy the original MTL to the output name
    if source_mtl.exists():
        shutil.copy(source_mtl, output_mtl_path)
        print(f"      -> Copied MTL to: {output_mtl_path.name}")
    else:
        print(f"      -> WARNING: Source MTL {source_mtl.name} not found. Output might lack material.")

    # 3. WRITE OUTPUT OBJ
    print(f"[3/4] Writing Output Mesh...")
    vertex_count = 0
    
    with open(target_path, 'r') as f_in, open(output_path, 'w') as f_out:
        # Write Header
        f_out.write(f"# Color Transfer by Script\n")
        
        for line in f_in:
            if line.startswith('v '):
                # It is a vertex line
                parts = line.strip().split()
                xyz = parts[1:4] # Keep Target Geometry (x, y, z)
                
                # Get Color from Source
                if vertex_count < len(colors):
                    rgb = colors[vertex_count]
                else:
                    rgb = ["0.5", "0.5", "0.5"] # Fallback Grey
                
                # Write combined: v x y z r g b
                f_out.write(f"v {' '.join(xyz)} {' '.join(rgb)}\n")
                vertex_count += 1
                
            elif line.startswith('mtllib'):
                # Update the library link to the new filename
                f_out.write(f"mtllib {output_mtl_name}\n")
                
            elif line.startswith('usemtl'):
                # Keep material usage lines
                f_out.write(line)
                
            else:
                # Keep faces (f), normals (vn), uvs (vt), and others intact
                f_out.write(line)

    # 4. VALIDATION
    print(f"[4/4] Complete.")
    if vertex_count != len(colors):
        print(f"      WARNING: Vertex count mismatch!")
        print(f"      Source Colors: {len(colors)}")
        print(f"      Target Verts:  {vertex_count}")
    else:
        print(f"      Success! {vertex_count} vertices colored.")
    
    print(f"      Saved to: {output_path}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Update these paths to your files
    SOURCE_OBJ = "/CT/SOMA/static00/S1/layers/apose/skin_layer_apose_baked.obj"
    TARGET_OBJ = "/CT/SOMA/static00/S5/layers/apose/skin_layer-S5-APose.obj"
    OUTPUT_OBJ = "/CT/SOMA/static00/S5/layers/apose/skin_layer-S5-APose_baked.obj"
    # ---------------------

    # Allow command line args
    if len(sys.argv) == 4:
        SOURCE_OBJ = sys.argv[1]
        TARGET_OBJ = sys.argv[2]
        OUTPUT_OBJ = sys.argv[3]

    if not os.path.exists(SOURCE_OBJ):
        print(f"Error: Source file not found: {SOURCE_OBJ}")
    else:
        transfer_vertex_colors_raw(SOURCE_OBJ, TARGET_OBJ, OUTPUT_OBJ)