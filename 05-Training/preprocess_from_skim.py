"""
preprocess_from_skim.py -- build the training-ready per-frame .npy dataset from the
pure-Python SKIM residual dataset (NPZ), instead of the old Blender JSON files.

Background: the original 00_preprocess_data.py reads Blender-exported per-shot JSON
(`*_residuals_*_world_lbs_scaled_tpose.json`, masks, LBS export). Those no longer exist for
the re-triangulated shots (S3/009-012) or S4, and the whole residual set was regenerated in
pure Python into /CT/SOMA/static00/SKIM_dataset/<S>/shot_*.npz. This adapter consumes that NPZ and
writes the EXACT same training layout the model loader expects.

Output (one dir per subject):
  <out>/<S>/preprocessed_vFinal_clean/{pose_rotations,residuals,masks,canonical_lbs}/shot_NNN_frame_FFFF.npy
    pose_rotations : (J*6,) float32  -- 6D pose, OLD convention (root Z-up fix; [col0|col1] block order)
    residuals      : (M,3)  float32  -- SKIM residual_scaled (= residual_m * 0.1), training's scaled GT
    masks          : (M,)   uint8    -- SKIM visibility mask
    canonical_lbs  : (M,3)  float32  -- marker-LBS world position (loaded by the loader, unused in loss)

Conventions / alignment (all verified):
  * Marker order in the NPZ == sorted(barycentric_map.keys()) == training order (identical, checked).
  * Residuals are the SCALED residual (existing training .npy used m*0.1).
  * Pose is reproduced FROM THE BVH with 00_preprocess_data's exact math (NOT the NPZ pose6d, whose
    column order/root frame differ). Output frame i uses BVH frame = NPZ frame_index[i] (the new
    pipeline's "output frame i <-> bvh frame i+1"), keeping pose and residual on the same bvh frame.

Run:
  /CT/SOMA/static00/miniforge3/envs/soma/bin/python preprocess_from_skim.py --subjects S1 S2 S3 S4 S5
  ... --validate-only            # just check S1 pose reproduction vs existing, write nothing
"""
import os, sys, glob, json, argparse
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/CT/SOMA/work/08-Residuals-Python")
from pymotion.io.bvh import BVH
import pymotion.rotations.quat as quat
import pymotion.rotations.ortho6d as sixd
import batch_residuals as B
from recreate_residuals import quat_wxyz_to_matrix, lbs_frame, rot_x, DEGREE_BVH_X

SKIM = "/CT/SOMA/static00/SKIM_dataset"
EXISTING = "/CT/SOMA/static00"          # for pose-reproduction validation only


def pose6d_old_all(bvh):
    """Per-bvh-frame (F, J*6) 6D pose in 00_preprocess_data.py's exact convention:
    root quaternion rotated by Z@Y@X(90 deg) into Z-up, sixd.from_quat, columns ordered [col0|col1]."""
    local_rotations, _, _, _, _, _ = bvh.get_data()
    local_rotations = local_rotations.copy()
    angles = np.array([0, np.pi / 2, 0])[..., np.newaxis]
    axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    rmf = quat.to_matrix(quat.from_angle_axis(angles, axes))
    R_total = rmf[2] @ rmf[1] @ rmf[0]
    q_root = local_rotations[:, 0, :]
    rotmats_root = Rotation.from_quat(np.roll(q_root, -1, axis=1)).as_matrix()
    R_new = R_total @ rotmats_root
    local_rotations[:, 0, :] = np.roll(Rotation.from_matrix(R_new).as_quat(), 1, axis=1)
    c6d = sixd.from_quat(local_rotations)                       # (F,J,3,2)
    rot_ordered = np.concatenate([c6d[..., 0], c6d[..., 1]], axis=-1)   # (F,J,6)
    return rot_ordered.reshape(rot_ordered.shape[0], -1).astype(np.float32)


def subject_model_for_lbs(S):
    """Reuse batch_residuals' subject model (canonical markers + weight matrix) for the
    (unused-but-required) canonical_lbs reconstruction."""
    return B.load_subject_model(S)


def lbs_world_all(S, shot, model, bvh, frame_index, gR, gt):
    """Marker-LBS world (Z-up) for each output frame, matching the SKIM pipeline
    (global_R/t applied). Only needed to fill canonical_lbs/*.npy."""
    lr, lp, parents, offsets, _, _ = bvh.get_data()
    parents = parents.copy(); parents[0] = -1
    J = len(parents)
    bscale = 0.001 if np.abs(offsets).max() > 10.0 else 1.0
    offsets = offsets * bscale
    root_all = lp[:, 0, :] * bscale
    j_rest = np.zeros_like(offsets)
    for i in range(J):
        j_rest[i] = offsets[i] if parents[i] < 0 else j_rest[parents[i]] + offsets[i]
    dom = model["W"].argmax(1)
    t_align = np.median(model["p_bind"] - j_rest[dom], axis=0)
    if np.linalg.norm(t_align) > 0.25:
        j_rest = j_rest + t_align
    p_bind, W = model["p_bind"], model["W"]
    out = np.zeros((len(frame_index), len(model["markers"]), 3), np.float32)
    for i, bf in enumerate(frame_index):
        rot = quat_wxyz_to_matrix(lr[int(bf)])
        d = (rot_x(-DEGREE_BVH_X) @ lbs_frame(p_bind, W, rot, j_rest, parents, root_all[int(bf)]).T).T
        out[i] = (gR @ d.T).T + gt
    return out


def process_subject(S, out_root, write=True, lbs=True, log=print):
    npzs = sorted(glob.glob(f"{SKIM}/{S}/shot_*.npz"))
    if not npzs:
        log(f"  {S}: no SKIM npz found, skip"); return 0
    model = subject_model_for_lbs(S) if (write and lbs) else None
    dirs = {k: f"{out_root}/{S}/preprocessed_vFinal_clean/{k}"
            for k in ("pose_rotations", "residuals", "masks", "canonical_lbs")}
    if write:
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)
    total = 0
    for npz in npzs:
        shot = os.path.basename(npz)[:-4]                  # shot_NNN
        num = shot.split("_")[1]
        d = np.load(npz, allow_pickle=True)
        res = d["residual_scaled"].astype(np.float32)      # (F,M,3) scaled GT
        mask = d["mask"].astype(np.uint8)                  # (F,M)
        fidx = d["frame_index"].astype(int)                # (F,) bvh frame per output frame
        gR, gt = d["global_R"].astype(np.float64), d["global_t"].astype(np.float64)
        F = res.shape[0]
        bvh = BVH(); bvh.load(B.bvh_path(S, shot))
        pose_all = pose6d_old_all(bvh)                     # (Nb, J*6)
        lbsw = lbs_world_all(S, shot, model, bvh, fidx, gR, gt) if (write and lbs) else None
        if write:
            for i in range(F):
                bf = int(fidx[i])
                fn = f"{shot}_frame_{i:04d}.npy"
                np.save(f"{dirs['pose_rotations']}/{fn}", pose_all[bf])
                np.save(f"{dirs['residuals']}/{fn}", res[i])
                np.save(f"{dirs['masks']}/{fn}", mask[i])
                np.save(f"{dirs['canonical_lbs']}/{fn}", lbsw[i])
        total += F
        log(f"  {S}/{shot}: F={F} (bvh {pose_all.shape[0]}) M={res.shape[1]} written={write}")
    log(f"  {S}: {len(npzs)} shots, {total} frames")
    return total


def validate_pose_S1():
    """Confirm pose reproduction matches the EXISTING S1 training data exactly.
    Existing S1 had no frame padding, so existing pose file frame k == pose_all[k]."""
    bvh = BVH(); bvh.load(B.bvh_path("S1", "shot_001"))
    pose_all = pose6d_old_all(bvh)
    errs = []
    for k in (0, 6, 100, 1000, 2000):
        p = f"{EXISTING}/S1/preprocessed_vFinal_clean/pose_rotations/shot_001_frame_{k:04d}.npy"
        if os.path.exists(p):
            errs.append((k, float(np.abs(np.load(p) - pose_all[k]).max())))
    print("[validate] S1 pose reproduction max-abs-diff vs existing per frame:")
    for k, e in errs:
        print(f"    frame {k:5d}: {e:.2e}")
    ok = all(e < 1e-5 for _, e in errs)
    print(f"[validate] {'OK - convention matches' if ok else 'MISMATCH'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=["S1", "S2", "S3", "S4", "S5"])
    ap.add_argument("--out", default=SKIM)
    ap.add_argument("--no-lbs", action="store_true", help="write zeros for canonical_lbs (it is unused in training)")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    if args.validate_only:
        validate_pose_S1(); return
    if not validate_pose_S1():
        print("ABORT: pose convention does not match existing data."); sys.exit(1)
    gt = 0
    for S in args.subjects:
        print(f"=== {S} ===")
        gt += process_subject(S, args.out, write=True, lbs=not args.no_lbs)
    print(f"DONE: {gt} frames total -> {args.out}/<S>/preprocessed_vFinal_clean")


if __name__ == "__main__":
    main()
