import os
import glob
import numpy as np
import trimesh
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import json

# --- CONFIGURATION ---
SUBJECT = "S1"
SEQUENCE = "shot_002"

NUMBER_OF_FRAMES = None

BASE_DIR = Path(rf"/CT/soma-experiments/work/HIT/outputs_soma_eval/{SUBJECT}/{SEQUENCE}")  # Adjust base path as needed
RESULTS_FILE = Path(f"hit_bio_evaluation_results_{SUBJECT}_{SEQUENCE}.json")
HIT_SUBFOLDER = ["hit_male_best", "hit_female_best"]

def calculate_intersection_ratio(skin_mesh, muscle_mesh):
    """
    Calculates the ratio of muscle vertices that are OUTSIDE the skin.
    0.0 = Perfect (All muscle inside skin)
    1.0 = Terrible (All muscle outside skin)
    """
    # 1. Get vertices of the muscle (Lean Tissue)
    muscle_points = muscle_mesh.vertices
    
    # 2. Check which points are inside the skin
    # trimesh.contains_points uses ray casting (Even-Odd rule)
    # It requires the mesh to be watertight-ish. 
    inside_mask = skin_mesh.contains(muscle_points)
    
    # 3. Calculate Ratio
    # We want the ratio of points *outside* (intersection/collision)
    num_outside = np.sum(~inside_mask)
    total_points = len(muscle_points)
    
    return num_outside / total_points

def safe_volume(mesh):
    """
    Calculates volume. Handles open meshes by attempting to close them 
    or using the surface integral (divergence theorem) which trimesh does by default.
    """
    if mesh.is_watertight:
        return mesh.volume
    else:
        # If open, trimesh still computes the surface integral.
        # It might be inaccurate if holes are large, but it's the best proxy for HIT.
        # We take absolute value because normal flips can cause negative volume.
        return abs(mesh.volume)

def save_state(data):
    """Overwrites the JSON file with the current data."""
    with open(RESULTS_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def load_state():
    """
    Robustly loads the JSON. 
    Guarantees returning a DICT {"summary": {}, "frames": [...]}.
    """
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, 'r') as f:
                data = json.load(f)
                
            # CASE 1: Data is a List (Legacy format) -> Convert it
            if isinstance(data, list):
                print(f"[INFO] Migrating legacy list format ({len(data)} frames) to new dict format.")
                return {"summary": {}, "frames": data}
            
            # CASE 2: Data is already a Dict -> Return it
            if isinstance(data, dict):
                # Ensure keys exist
                if "frames" not in data: data["frames"] = []
                if "summary" not in data: data["summary"] = {}
                return data
                
        except json.JSONDecodeError:
            print("[WARN] JSON file corrupt or empty. Starting fresh.")
            return {"summary": {}, "frames": []}
            
    # CASE 3: No file -> New Dict
    return {"summary": {}, "frames": []}

def run_evaluation():
    # 1. Load existing state (Guaranteed to be a Dict now)
    full_data = load_state()
    
    # Double check structure to prevent errors
    if isinstance(full_data, list):
        full_data = {"summary": {}, "frames": full_data}
    if "frames" not in full_data: full_data["frames"] = []
    if "summary" not in full_data: full_data["summary"] = {}

    # Create a lookup set for processed frames
    processed_frames = {entry['frame'] for entry in full_data['frames']}
    print(f"Loaded {len(full_data['frames'])} existing entries.")

    # 2. Get all frame directories
    all_frame_dirs = sorted(glob.glob(os.path.join(BASE_DIR, "frame_*")))
    
    # Optional: Test with slice, e.g., all_frame_dirs[:15]
    if NUMBER_OF_FRAMES:
        frame_dirs = all_frame_dirs[:NUMBER_OF_FRAMES]
    else:
        frame_dirs = all_frame_dirs

    print(f"Found {len(frame_dirs)} frames in {BASE_DIR}")
    
    # 3. Processing Loop
    for f_dir in tqdm(frame_dirs, desc="Evaluating"):
        frame_name = os.path.basename(f_dir)
        
        if frame_name in processed_frames:
            continue
        
        # target_dir = os.path.join(f_dir, HIT_SUBFOLDER)
        target_dir = None
        for sub in HIT_SUBFOLDER:
            candidate = os.path.join(f_dir, sub)
            if os.path.isdir(candidate):
                target_dir = candidate
                break
        
        if target_dir is None:
            # print(f"Skipping {frame_name}: No 'hit_male_best' or 'hit_female_best' found.")
            continue

        path_skin = os.path.join(target_dir, "smpl_mesh.obj")
        path_lean = os.path.join(target_dir, "LT_mesh.obj") 
        path_fat  = os.path.join(target_dir, "AT_mesh.obj") 
        
        if not all(os.path.exists(p) for p in [path_skin, path_lean, path_fat]):
            continue
            
        try:
            # Measure
            mesh_skin = trimesh.load(path_skin, process=True)
            mesh_lean = trimesh.load(path_lean, process=True)
            mesh_fat  = trimesh.load(path_fat, process=True)
            
            int_ratio = calculate_intersection_ratio(mesh_skin, mesh_lean)
            v_fat = safe_volume(mesh_fat)
            v_mus = safe_volume(mesh_lean)
            
            # Append
            new_entry = {
                "frame": frame_name,
                "intersection_ratio": float(int_ratio), 
                "vol_fat": float(v_fat),
                "vol_muscle": float(v_mus)
            }
            
            # Append to the 'frames' list, not the dict
            full_data['frames'].append(new_entry)
            
            # Save immediately
            save_state(full_data) 
            
        except Exception as e:
            print(f"Error processing {frame_name}: {e}")

    # 4. FINAL SUMMARY CALCULATION
    if len(full_data['frames']) > 0:
        # [FIX] Create DataFrame from the LIST of frames, not the whole DICT
        df = pd.DataFrame(full_data['frames'])
        
        # Stats
        mean_int = df["intersection_ratio"].mean()
        max_int = df["intersection_ratio"].max()
        
        fat_mean = df["vol_fat"].mean()
        fat_cv = (df["vol_fat"].std() / fat_mean) if fat_mean > 0 else 0.0
        
        mus_mean = df["vol_muscle"].mean()
        mus_cv = (df["vol_muscle"].std() / mus_mean) if mus_mean > 0 else 0.0
        
        # Update Summary in Dict
        full_data['summary'] = {
            "intersection_ratio_mean": float(mean_int),
            "intersection_ratio_max": float(max_int),
            "volume_cv_fat": float(fat_cv),
            "volume_cv_muscle": float(mus_cv),
            "total_frames": len(df)
        }
        
        print("\n=== EVALUATION REPORT ===")
        print(f"Intersection Ratio (Lower is better):")
        print(f"  Mean: {mean_int:.4f} ({mean_int*100:.2f}%)")
        print(f"  Max: {max_int:.4f} ({max_int*100:.2f}%)") 

        print(f"\nVolume Preservation (CV = Std/Mean):")
        print(f"  Fat CV:    {fat_cv:.4f} ({fat_cv*100:.2f}%)")
        print(f"  Muscle CV: {mus_cv:.4f} ({mus_cv*100:.2f}%)")
        
        # Final Save
        save_state(full_data)

        print("\n=== FINAL RESULTS SAVED TO JSON ===")
        print(json.dumps(full_data['summary'], indent=4))

        # Plotting
        try:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            df_sorted = df.sort_values("frame") 
            plt.plot(df_sorted["intersection_ratio"], label="LT outside SMPL")
            plt.title(f"Intersection (Mean: {mean_int:.4f})")
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(df_sorted["vol_fat"], label="Fat Vol")
            plt.plot(df_sorted["vol_muscle"], label="Muscle Vol")
            plt.title(f"Volume Stability (Fat CV: {fat_cv:.4f})")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.savefig(f"hit_bio_evaluation_plot_{SUBJECT}_{SEQUENCE}.png")
            print(f"Plot saved.")
        except Exception as e:
            print(f"Plotting failed: {e}")

if __name__ == "__main__":
    run_evaluation()