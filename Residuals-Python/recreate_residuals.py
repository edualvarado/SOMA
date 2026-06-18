"""
recreate_residuals.py  --  pure-Python (no Blender) recreation of marker residuals.

Pipeline (method A: direct marker-LBS, consistent with 05-Training):
  1. Load canonical tpose markers (bind pose) and rotate -90deg X into the BVH frame.
  2. Build a dense (M, 24) marker LBS weight matrix from the exported marker weights.
  3. For each BVH frame, LBS-deform the canonical markers (lbs_working_batch_rotmat math),
     then rotate +90deg X back to the Z-up world frame.
  4. residual = observed_triangulation - LBS_marker   (world space)
     masked   = 1 if the marker was observed that frame else 0
     scaled   = residual * SCALE_FACTOR
  5. Compare against the existing Blender-made residual / mask files.

Default run = validation/comparison on a sample of frames (fast, no big files written).
Use --full to evaluate every frame; --dump to also write the recreated residual JSON.

    /CT/SOMA/static00/miniforge3/envs/soma/bin/python recreate_residuals.py
"""
import os
import json
import argparse
import numpy as np
from pymotion.io.bvh import BVH

# ----------------------------- config ------------------------------------
SUBJECT = "S1"
SHOT = "shot_001"
BASE = f"/CT/SOMA/static00/{SUBJECT}"
SHOT_DIR = f"{BASE}/raw/{SHOT}_captury"

BVH_PATH = f"{SHOT_DIR}/{SUBJECT}_{SHOT}.bvh"
CANON_PATH = f"{BASE}/canonical_model/{SUBJECT}_canonical_data_tpose.json"
BARY_PATH = f"{BASE}/canonical_model/generated_marker_barycentric_map.json"
WEIGHTS_PATH = f"{BASE}/canonical_model/lbs_markers/{SUBJECT}_marker_lbs_weights_exported.json"
TRIANG_PATH = f"{SHOT_DIR}/{SUBJECT}_triangulated_sequence_{SHOT}_transformed_filtered.json"

EXIST_RESID = f"{SHOT_DIR}/{SUBJECT}_residuals_{SHOT}_world_lbs_scaled_tpose.json"
EXIST_MASK = f"{SHOT_DIR}/{SUBJECT}_masked_residuals_{SHOT}_world_tpose.json"

DEGREE_BVH_X = -90.0   # canonical (Z-up) -> BVH (Y-up); same constant as 05-Training
# NOTE: the committed 15_scale_residuals.py says 0.01, but the existing
# S1_residuals_*_scaled_tpose.json files were actually produced with 0.1
# (verified exactly: existing == (triangulation - blender_export) * 0.1, rmse 0).
SCALE_FACTOR = 0.1
# -------------------------------------------------------------------------


def rot_x(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def quat_wxyz_to_matrix(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = w * w + x * x + y * y + z * z
    s = np.where(n > 0, 2.0 / n, 0.0)
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    m = np.empty(q.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1 - (yy + zz); m[..., 0, 1] = xy - wz;     m[..., 0, 2] = xz + wy
    m[..., 1, 0] = xy + wz;     m[..., 1, 1] = 1 - (xx + zz); m[..., 1, 2] = yz - wx
    m[..., 2, 0] = xz - wy;     m[..., 2, 1] = yz + wx;     m[..., 2, 2] = 1 - (xx + yy)
    return m


def rest_joints(offsets, parents, zero_root_offset):
    offsets = offsets.copy()
    if zero_root_offset:
        offsets[0] = 0.0
    J = len(parents)
    jr = np.zeros_like(offsets)
    for i in range(J):
        jr[i] = offsets[i] if parents[i] < 0 else jr[parents[i]] + offsets[i]
    return jr


def lbs_frame(v_bind, W, rot_mats, j_rest, parents, root_pos):
    """v_bind (N,3), rot_mats (J,3,3) -> deformed (N,3); all in BVH frame.
    Identical math to 05-Training lbs_working_batch_rotmat."""
    J = len(parents)
    G_rest = np.tile(np.eye(4), (J, 1, 1))
    for i in range(J):
        G_rest[i, :3, 3] = j_rest[i]
    G_posed = np.zeros((J, 4, 4))
    for i in range(J):
        L = np.eye(4)
        L[:3, :3] = rot_mats[i]
        if parents[i] < 0:
            L[:3, 3] = root_pos
            G_posed[i] = L
        else:
            L[:3, 3] = j_rest[i] - j_rest[parents[i]]
            G_posed[i] = G_posed[parents[i]] @ L
    skin = G_posed @ np.linalg.inv(G_rest)
    blended = np.einsum('vj,jmn->vmn', W, skin)
    homo = np.concatenate([v_bind, np.ones((len(v_bind), 1))], axis=1)
    return np.einsum('vmn,vn->vm', blended, homo)[:, :3]


def load_model(zero_root_offset):
    bvh = BVH(); bvh.load(BVH_PATH)
    lr, lp, parents, offsets, _, _ = bvh.get_data()
    parents = parents.copy(); parents[0] = -1
    j_rest = rest_joints(offsets, parents, zero_root_offset)

    canon = json.load(open(CANON_PATH))["0"]
    bary = json.load(open(BARY_PATH))
    weights = json.load(open(WEIGHTS_PATH))
    # marker order = sorted bary keys (same subset/order 05-Training uses), present in all sources
    markers = [m for m in sorted(bary.keys()) if m in canon and m in weights]
    p_canon = np.array([canon[m][0] for m in markers])            # Z-up
    p_bind = (rot_x(DEGREE_BVH_X) @ p_canon.T).T                  # BVH frame
    W = np.zeros((len(markers), len(parents)))
    for r, m in enumerate(markers):
        wi = weights[m]
        for bi, w in zip(wi["bone_indices"], wi["weights"]):
            W[r, bi] = w
    return lr, lp, parents, j_rest, markers, p_bind, W


def lbs_markers_world(f, lr, lp, parents, j_rest, p_bind, W):
    """Posed markers for bvh frame f, returned in Z-up world frame."""
    rot = quat_wxyz_to_matrix(lr[f])
    d = lbs_frame(p_bind, W, rot, j_rest, parents, lp[f, 0, :])
    return (rot_x(-DEGREE_BVH_X) @ d.T).T                          # back to Z-up


def write_full_npz(args):
    """Generate the complete per-frame output for the shot and save a compact .npz.

    Frame f (0..nfr-1) is indexed by BVH frame; observation comes from triangulation[str(f)].
    Triangulation and the existing residuals are both keyed 0..2347, so the alignment is 1:1
    (an unobserved marker in a frame gets residual 0 + mask 0).
    Arrays (float32 unless noted), marker order = markers:
        residual_scaled (F,M,3)  = (observed - LBS) * SCALE_FACTOR   (0 where unobserved)
        residual_m      (F,M,3)  = observed - LBS, in metres         (0 where unobserved)
        lbs_world       (F,M,3)  = posed canonical markers (Z-up world)
        mask            (F,M) u8 = 1 if observed that frame else 0
    """
    lr, lp, parents, j_rest, markers, p_bind, W = load_model(args.zero_root_offset)
    nfr = lr.shape[0]; M = len(markers)
    midx = {m: i for i, m in enumerate(markers)}
    print(f"[full] {SUBJECT}/{SHOT}: {nfr} frames x {M} markers, zero_root_offset={args.zero_root_offset}")
    tri = json.load(open(TRIANG_PATH))

    residual_m = np.zeros((nfr, M, 3), np.float32)
    lbs_world = np.zeros((nfr, M, 3), np.float32)
    mask = np.zeros((nfr, M), np.uint8)
    for f in range(nfr):
        lbsw = lbs_markers_world(f, lr, lp, parents, j_rest, p_bind, W)
        lbs_world[f] = lbsw
        obs = tri.get(str(f), {})
        for m, val in obs.items():
            i = midx.get(m)
            if i is not None:
                residual_m[f, i] = np.array(val[0]) - lbsw[i]
                mask[f, i] = 1
        if f % 400 == 0:
            print(f"  frame {f}/{nfr}", end="\r")
    residual_scaled = (residual_m * SCALE_FACTOR).astype(np.float32)
    np.savez_compressed(args.npz, marker_ids=np.array(markers), frame_index=np.arange(nfr),
                        residual_scaled=residual_scaled, residual_m=residual_m,
                        lbs_world=lbs_world, mask=mask,
                        scale_factor=np.float32(SCALE_FACTOR))
    obs_res = residual_m[mask.astype(bool)]
    print(f"\n[full] observed marker-frames: {int(mask.sum())}; "
          f"residual mean={np.linalg.norm(obs_res,axis=1).mean()*100:.2f}cm")
    print(f"[full] wrote {args.npz} ({os.path.getsize(args.npz)/1e6:.1f} MB)")

    if args.dump:
        out = {}
        for f in range(nfr):
            fo = {}
            for m in markers:
                i = midx[m]
                fo[m] = [residual_scaled[f, i].tolist()] if mask[f, i] else [[0.0, 0.0, 0.0]]
            out[str(f)] = fo
        with open(args.dump, "w") as fh:
            json.dump(out, fh)
        print(f"[full] wrote JSON {args.dump} ({os.path.getsize(args.dump)/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="evaluate every frame")
    ap.add_argument("--dump", metavar="PATH", help="write recreated scaled-residual JSON")
    ap.add_argument("--npz", metavar="PATH",
                    help="write the full per-frame output (residual/mask/lbs) as a compressed .npz")
    ap.add_argument("--zero-root-offset", action="store_true",
                    help="zero the root offset in j_rest (matches 05-Training)")
    args = ap.parse_args()
    if args.npz:
        return write_full_npz(args)

    lr, lp, parents, j_rest, markers, p_bind, W = load_model(args.zero_root_offset)
    nfr = lr.shape[0]
    midx = {m: i for i, m in enumerate(markers)}
    print(f"[setup] {SUBJECT}/{SHOT}: {nfr} bvh frames, {len(markers)} markers, "
          f"zero_root_offset={args.zero_root_offset}")

    print("[load] triangulation + existing residual ...")
    tri = json.load(open(TRIANG_PATH))
    exist = json.load(open(EXIST_RESID))

    # frame alignment: triangulation key str(f) <-> bvh frame f (residual["0"]=zeros)
    frames = range(1, nfr) if args.full else range(100, nfr, 200)

    vs_obs, vs_exist, dump = [], [], {}
    for f in frames:
        tkey = str(f)
        if tkey not in tri:
            continue
        obs = tri[tkey]
        lbsw = lbs_markers_world(f, lr, lp, parents, j_rest, p_bind, W)
        ekey = str(f)
        ex = exist.get(ekey, {})
        frame_out = {}
        for m in markers:
            i = midx[m]
            if m in obs:
                res = np.array(obs[m][0]) - lbsw[i]          # world residual
                vs_obs.append(np.linalg.norm(res))
                frame_out[m] = [(res * SCALE_FACTOR).tolist()]
                if m in ex:                                   # compare scaled residual
                    vs_exist.append(np.linalg.norm(res * SCALE_FACTOR - np.array(ex[m])))
            else:
                frame_out[m] = [[0.0, 0.0, 0.0]]
        if args.dump:
            dump[ekey] = {m: v for m, v in frame_out.items()}

    vs_obs = np.array(vs_obs); vs_exist = np.array(vs_exist)
    print(f"\n[my residual magnitude] |observed - myLBS| over {len(vs_obs)} marker-frames:")
    print(f"   mean={vs_obs.mean()*100:.2f}cm  median={np.median(vs_obs)*100:.2f}cm  "
          f"p95={np.percentile(vs_obs,95)*100:.2f}cm")
    # convert the scaled-unit difference back to real cm via /SCALE_FACTOR
    diff_cm = vs_exist / SCALE_FACTOR * 100
    print(f"[my residual vs existing] difference between method A (marker-LBS) and the "
          f"existing Blender (skin-mesh) residual, over {len(vs_exist)} marker-frames:")
    print(f"   mean={diff_cm.mean():.2f}cm  median={np.median(diff_cm):.2f}cm  "
          f"p95={np.percentile(diff_cm,95):.2f}cm  (= |export - myLBS|, the method gap)")

    if args.dump:
        with open(args.dump, "w") as fo:
            json.dump(dump, fo)
        print(f"\n[dump] wrote {args.dump} ({len(dump)} frames)")


if __name__ == "__main__":
    main()
