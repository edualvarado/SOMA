import json
import os
import shutil
import numpy as np

def transfer_barycentric_map(source_map_path, target_map_path):
    print(f"[TRANSFER] Loading Source Map: {source_map_path}")
    
    if not os.path.exists(source_map_path):
        print(f"[ERROR] Source map not found! Generate S1 map first.")
        return

    with open(source_map_path, 'r') as f:
        source_data = json.load(f)

    print(f"           Found {len(source_data)} markers in source.")
    
    # We essentially just copy the file, but we can filter/validate if needed.
    # For shared topology, a direct copy is usually the best starting point.
    # It ensures S2 has the EXACT same marker set and face associations as S1.
    
    print(f"[TRANSFER] Saving to Target: {target_map_path}")
    
    # Create directory if missing
    os.makedirs(os.path.dirname(target_map_path), exist_ok=True)
    
    with open(target_map_path, 'w') as f:
        json.dump(source_data, f, indent=4)
        
    print(f"[SUCCESS] Transferred map. S2 now uses S1's topology.")

def main():
    # --- CONFIGURATION ---
    # We want to use S1 (which was working well) as the Master
    SOURCE_SUBJECT = "S1"
    TARGET_SUBJECT = "S2" # Change this for S3, S5, etc.
    
    BASE_ROOT = "/CT/SOMA/static00"
    
    source_path = os.path.join(BASE_ROOT, SOURCE_SUBJECT, "canonical_model", "generated_marker_barycentric_map.json")
    target_path = os.path.join(BASE_ROOT, TARGET_SUBJECT, "canonical_model", "generated_marker_barycentric_map_new.json")
    
    print(f"--- MAP TOPOLOGY TRANSFER ---")
    print(f"Source: {SOURCE_SUBJECT}")
    print(f"Target: {TARGET_SUBJECT}")
    
    transfer_barycentric_map(source_path, target_path)

if __name__ == "__main__":
    main()