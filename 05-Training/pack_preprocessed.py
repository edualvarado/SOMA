"""
pack_preprocessed.py -- pack the per-frame training .npy into one compressed .npz per shot,
to make the SKIM dataset light to distribute (1.14M tiny files -> 74 files).

Each packed file  <S>/preprocessed_vFinal_clean/<shot>.npz  holds the whole shot:
    pose       (F, J*6) float32   -- 6D pose (same convention as the per-frame pose_rotations)
    residuals  (F, M, 3) float16  -- scaled GT residual (residual_m * 0.1); cast to float at load
    masks      (F, M)   uint8     -- visibility
The unused-in-training `canonical_lbs` is intentionally dropped (the loader synthesizes a zero
placeholder so the existing 4-tuple training interface is unchanged).

It is regenerated from the source (SKIM <shot>.npz + the shot BVH) using the SAME math as
preprocess_from_skim.py -- identical values to the per-frame files (verified by --validate),
so it does not need to read the 1.14M per-frame .npy.

Run:
  /CT/SOMA/static00/miniforge3/envs/soma/bin/python pack_preprocessed.py --subjects S1 S2 S3 S4 S5
  ... --validate     # additionally compare packed vs the existing per-frame .npy and report
"""
import os, sys, glob, argparse
import numpy as np

sys.path.insert(0, "/CT/SOMA/work/08-Residuals-Python")
from pymotion.io.bvh import BVH
import batch_residuals as B
from preprocess_from_skim import pose6d_old_all, SKIM


def pack_subject(S, root, log=print):
    npzs = sorted(glob.glob(f"{SKIM}/{S}/shot_*.npz"))
    out_dir = f"{root}/{S}/preprocessed_vFinal_clean"
    os.makedirs(out_dir, exist_ok=True)
    total = 0
    for npz in npzs:
        shot = os.path.basename(npz)[:-4]
        d = np.load(npz, allow_pickle=True)
        res = d["residual_scaled"].astype(np.float16)   # (F,M,3) scaled GT
        mask = d["mask"].astype(np.uint8)               # (F,M)
        fidx = d["frame_index"].astype(int)             # (F,)
        bvh = BVH(); bvh.load(B.bvh_path(S, shot))
        pose = pose6d_old_all(bvh)[fidx].astype(np.float32)   # (F, J*6)
        np.savez_compressed(f"{out_dir}/{shot}.npz", pose=pose, residuals=res, masks=mask)
        total += res.shape[0]
        log(f"  {S}/{shot}: F={res.shape[0]} M={res.shape[1]} -> {os.path.getsize(f'{out_dir}/{shot}.npz')/1e6:.1f}MB")
    log(f"  {S}: {len(npzs)} shots, {total} frames packed")
    return total


def validate_subject(S, root, n_frames=6, log=print):
    """Compare a sample of packed frames against the existing per-frame .npy (if present)."""
    out_dir = f"{root}/{S}/preprocessed_vFinal_clean"
    pf_pose = f"{out_dir}/pose_rotations"
    if not os.path.isdir(pf_pose):
        log(f"  {S}: no per-frame dir to validate against (skip)"); return True
    ok = True
    for shot_npz in sorted(glob.glob(f"{out_dir}/shot_*.npz"))[:3]:
        shot = os.path.basename(shot_npz)[:-4]
        pk = np.load(shot_npz)
        F = pk["pose"].shape[0]
        for i in sorted(set(np.linspace(0, F - 1, n_frames).astype(int))):
            fn = f"{shot}_frame_{i:04d}.npy"
            pose_pf = np.load(f"{out_dir}/pose_rotations/{fn}")
            res_pf = np.load(f"{out_dir}/residuals/{fn}")
            mask_pf = np.load(f"{out_dir}/masks/{fn}")
            e_pose = np.abs(pk["pose"][i] - pose_pf).max()
            e_res = np.abs(pk["residuals"][i].astype(np.float32) - res_pf).max()
            e_mask = np.abs(pk["masks"][i].astype(np.int16) - mask_pf.astype(np.int16)).max()
            if not (e_pose < 1e-6 and e_res < 1e-6 and e_mask == 0):
                ok = False
                log(f"  !! {S}/{shot} frame {i}: dpose={e_pose:.2e} dres={e_res:.2e} dmask={e_mask}")
    log(f"  {S}: packed-vs-perframe {'OK' if ok else 'MISMATCH'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=["S1", "S2", "S3", "S4", "S5"])
    ap.add_argument("--out", default=SKIM)
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    allok = True
    for S in args.subjects:
        print(f"=== {S} ===")
        pack_subject(S, args.out)
        if args.validate:
            allok &= validate_subject(S, args.out)
    print("DONE" + ("" if allok else "  (VALIDATION MISMATCH!)"))


if __name__ == "__main__":
    main()
