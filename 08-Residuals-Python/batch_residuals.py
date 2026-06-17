"""
batch_residuals.py -- regenerate marker residuals for the whole SKIM dataset (S1-S5),
pure Python (no Blender), method-A direct marker-LBS, into a self-contained database.

Per (subject, shot):
  observed markers  = transform( triangulation_markers_processed.json )  [ -x, z, y ; "500"->marker_500_0_c ]
  LBS markers       = canonical markers posed by the shot BVH (marker LBS weights)
  residual_m        = observed - LBS        (metres, world Z-up; 0 where unobserved)
  residual_scaled   = residual_m * 0.1      (historical convention; both stored)
  mask              = 1 where the marker was observed that frame

Frame alignment (validated on S1): output frame i  <->  bvh frame i+1  <->  proc[i].

Output tree (default /CT/SOMA/static00/SKIM_dataset):
  metadata.json, README.md
  <S>/canonical.npz                 marker_ids, p_canon, p_bind, W, parents, joint_names
  <S>/<shot>.npz                    residual_m(f16), residual_scaled(f16), mask(u8),
                                    pose6d(f32 F,J,6), root_position(f32 F,3), j_rest(f32 J,3),
                                    frame_index(=bvh frame), marker_ids

Run (background):
  /CT/SOMA/static00/miniforge3/envs/soma/bin/python batch_residuals.py            # all subjects
  ... batch_residuals.py --subjects S1 --validate                                 # S1 + compare to existing
"""
import os, gc, json, glob, time, argparse
import numpy as np
from pymotion.io.bvh import BVH

from recreate_residuals import (quat_wxyz_to_matrix, lbs_frame, rot_x,
                                 DEGREE_BVH_X, SCALE_FACTOR)

STATIC = "/CT/SOMA/static00"
DATA = f"{STATIC}/data"
OUT_DEFAULT = "/CT/SOMA/static00/SKIM_dataset"

# subject -> data dir name; bvh path builder differs for S1
SUBJECTS = {
    "S1": "S1-21-11-24",
    "S2": "Valentin-12-12-25",
    "S3": "Timothee-17-12-25",
    "S4": "Mona-22-12-25",
    "S5": "Sarah-19-12-25",
}


def bvh_path(S, sh):
    if S == "S1":
        return f"{STATIC}/S1/raw/{sh}_captury/S1_{sh}.bvh"
    # Prefer the tracking_backup BVH: for some shots the main unknown.bvh was re-exported with a
    # changed/broken convention (~478 cm off). Where no backup exists, fall back to the main BVH --
    # for those shots (e.g. S3/005,008) it is the original and validates fine (~4-6 cm residual).
    bak = f"{DATA}/{SUBJECTS[S]}/{sh}_captury/tracking_backup/unknown.bvh"
    return bak if os.path.exists(bak) else f"{DATA}/{SUBJECTS[S]}/{sh}_captury/unknown.bvh"


def tri_path(S, sh):
    return (f"{DATA}/{SUBJECTS[S]}/{sh}/tracking_charuco-suit/triangulation/"
            f"3D-interpolated-N10/triangulation_markers_processed.json")


def shots_of(S):
    base = f"{DATA}/{SUBJECTS[S]}"
    out = []
    for d in sorted(glob.glob(f"{base}/shot_*")):
        sh = os.path.basename(d)
        if sh.endswith("_captury") or not sh[-1].isdigit():
            continue
        out.append(sh)
    return out


def canon_paths(S):
    cm = f"{STATIC}/{S}/canonical_model"
    return (f"{cm}/{S}_canonical_data_tpose.json",
            f"{cm}/generated_marker_barycentric_map.json",
            f"{cm}/lbs_markers/{S}_marker_lbs_weights_exported.json")


_FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def map_bone(name, bvh_idx):
    """Map a weight bone name onto the 24-joint BVH. S2-S5 weights are on the full 54-bone rig
    (fingers/eyes); the shot BVH only has body joints, so collapse fingers->hand, eyes->Head."""
    if name in bvh_idx:
        return bvh_idx[name]
    if name.startswith(_FINGERS):
        return bvh_idx["LeftHand"] if name.endswith("_L") else bvh_idx["RightHand"]
    if name.startswith("eye"):
        return bvh_idx["Head"]
    raise KeyError(f"unmapped bone {name!r}")


def load_relabel(S):
    """Optional proc_ID -> canonical_ID relabeling (derived by derive_relabel.py).
    Needed for S4/S5, whose triangulation uses a different marker-ID convention than
    the canonical model. Returns {} when no map exists (S1-S3: IDs already match)."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"relabel_{S}.json")
    if not os.path.exists(p):
        return {}
    return json.load(open(p))["map"]


def load_subject_model(S):
    """Per-subject, shot-independent: marker order, canonical/bind positions, weight matrix.
    Joint count/order taken from a reference shot's BVH (consistent across that subject)."""
    canon_p, bary_p, w_p = canon_paths(S)
    canon = json.load(open(canon_p))["0"]
    bary = json.load(open(bary_p))
    weights = json.load(open(w_p))
    markers = [m for m in sorted(bary.keys()) if m in canon and m in weights]
    ref_bvh = BVH(); ref_bvh.load(bvh_path(S, shots_of(S)[0]))
    joint_names = [str(n) for n in ref_bvh.data["names"]]
    bvh_idx = {n: i for i, n in enumerate(joint_names)}
    J = len(joint_names)
    p_canon = np.array([canon[m][0] for m in markers], np.float64)
    p_bind = (rot_x(DEGREE_BVH_X) @ p_canon.T).T
    W = np.zeros((len(markers), J))
    for r, m in enumerate(markers):
        wi = weights[m]
        for bn, w in zip(wi["bone_names"], wi["weights"]):   # map by NAME (index space varies per subject)
            W[r, map_bone(bn, bvh_idx)] += w
    # mid -> {corner: [(instance, marker_index), ...]}.  The suit reuses marker IDs, so the
    # canonical has instance-0 and instance-1 entries for some IDs while the triangulation always
    # labels observations as instance 0; each observed corner is later assigned to the instance
    # whose LBS prediction is closest (the "swap" the old 06_*_filter step performed).
    midx = {m: i for i, m in enumerate(markers)}
    mid_map = {}
    for m, i in midx.items():
        parts = m.split("_")            # marker_<id>_<instance>_<corner>
        mid, inst, c = parts[1], int(parts[2]), int(parts[3])
        mid_map.setdefault(mid, {}).setdefault(c, []).append((inst, i))
    return dict(markers=markers, midx=midx, mid_map=mid_map, relabel=load_relabel(S), p_canon=p_canon,
                p_bind=p_bind, W=W, joint_names=joint_names)


def rotmat_to_6d(R):
    """R (...,3,3) -> 6d (...,6): first two columns flattened."""
    return R[..., :, :2].reshape(R.shape[:-2] + (6,))


# observed-marker coordinate transform: [x,y,z] -> [-x, z, y]
_TF = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], np.float64)


def umeyama_rigid(X, Y):
    """Robust rigid (rotation+translation) transform mapping X->Y (rows = points)."""
    def fit(X, Y):
        mx, my = X.mean(0), Y.mean(0)
        U, D, Vt = np.linalg.svd((Y - my).T @ (X - mx) / len(X))
        S = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[2, 2] = -1
        R = U @ S @ Vt
        return R, my - R @ mx
    R, t = fit(X, Y)
    for _ in range(3):
        e = np.linalg.norm((R @ X.T).T + t - Y, axis=1)
        k = e < max(3 * np.median(e), 0.05)
        if k.sum() < 10:
            break
        R, t = fit(X[k], Y[k])
    return R, t


def process_shot(S, sh, model, out_dir, log, drop_thresh=None):
    out_npz = f"{out_dir}/{S}/{sh}.npz"
    if os.path.exists(out_npz):
        log(f"  {S}/{sh}: exists, skip"); return "skip"
    bvp, trp = bvh_path(S, sh), tri_path(S, sh)
    if not (os.path.exists(bvp) and os.path.exists(trp)):
        log(f"  {S}/{sh}: MISSING bvh={os.path.exists(bvp)} tri={os.path.exists(trp)}; skip"); return "missing"

    bvh = BVH(); bvh.load(bvp)
    lr, lp, parents, offsets, _, _ = bvh.get_data()
    parents = parents.copy(); parents[0] = -1
    Nb, J = lr.shape[0], len(parents)
    # Some BVHs (S2-S5) are in millimetres; canonical markers/observations are in metres. Scale to metres.
    bscale = 0.001 if np.abs(offsets).max() > 10.0 else 1.0
    offsets = offsets * bscale
    root_all = lp[:, 0, :] * bscale
    j_rest = np.zeros_like(offsets)
    for i in range(J):
        j_rest[i] = offsets[i] if parents[i] < 0 else j_rest[parents[i]] + offsets[i]

    # Re-register the rest skeleton to the canonical markers when the BVH root convention leaves them
    # grossly misaligned. S2-S5 store the hip height in the per-frame root channel with a zero rest
    # offset, so the rest skeleton sits at the origin while the canonical markers sit at body height
    # (median marker->bone offset ~0.82 m). S1 already aligns (offset ~0.08 m = real skin thickness),
    # so only correct when gross (>0.25 m), to avoid shifting S1 off its validated result.
    dom = model["W"].argmax(1)
    t_align = np.median(model["p_bind"] - j_rest[dom], axis=0)
    if np.linalg.norm(t_align) > 0.25:
        j_rest = j_rest + t_align
        log(f"  {S}/{sh}: re-aligned rest skeleton by {t_align.round(2)} (|t|={np.linalg.norm(t_align):.2f}m)")

    proc = json.load(open(trp))
    if not proc:                                   # empty triangulation (no observed frames)
        log(f"  {S}/{sh}: empty triangulation; skip"); return "missing"
    Np = max(int(k) for k in proc) + 1
    F = min(Np, Nb - 1)                       # output frame i <-> bvh frame i+1 <-> proc[i]

    markers = model["markers"]; mid_map = model["mid_map"]; relabel = model["relabel"]
    p_bind, W = model["p_bind"], model["W"]
    M = len(markers)

    def _lbsw(bf):
        rot = quat_wxyz_to_matrix(lr[bf])
        return (rot_x(-DEGREE_BVH_X) @ lbs_frame(p_bind, W, rot, j_rest, parents, root_all[bf]).T).T

    # Per-shot global rigid alignment of LBS -> observed: removes any residual coordinate offset
    # between the BVH world and the triangulation world (negligible for S1; ~6 cm for S2/S3/S5).
    gX, gY = [], []
    for i in range(min(200, max(F - 1, 1)), F, max(1, F // 30)):
        lw = _lbsw(i + 1); pf = proc.get(str(i), {})
        for mid, corners in pf.items():
            entry = relabel.get(mid) if relabel else {"canon": mid, "perm": (0, 1, 2, 3)}
            if entry is None:
                continue
            cmap = mid_map.get(entry["canon"])
            if not cmap:
                continue
            arr = (_TF @ np.asarray(corners, np.float64).T).T
            for c in range(len(corners)):
                cands = cmap.get(entry["perm"][c])
                if cands:
                    gX.append(lw[cands[0][1]]); gY.append(arr[c])
    gR, gt = umeyama_rigid(np.asarray(gX), np.asarray(gY)) if len(gX) > 100 else (np.eye(3), np.zeros(3))

    residual_m = np.zeros((F, M, 3), np.float32)
    mask = np.zeros((F, M), np.uint8)
    pose6d = np.zeros((F, J, 6), np.float32)
    root_pos = np.zeros((F, 3), np.float32)
    bvh_idx = np.zeros(F, np.int32)

    for i in range(F):
        bf = i + 1
        rot = quat_wxyz_to_matrix(lr[bf])                 # (J,3,3)
        lbsw = (gR @ _lbsw(bf).T).T + gt
        pf = proc.get(str(i), {})
        for mid, corners in pf.items():
            # subjects with a relabel map (S4/S5): translate the proc ID and corner order;
            # IDs absent from the map are untrusted and dropped.
            if relabel:
                entry = relabel.get(mid)
                if entry is None:
                    continue
                cid, perm = entry["canon"], entry["perm"]
            else:
                cid, perm = mid, (0, 1, 2, 3)
            cmap = mid_map.get(cid)
            if not cmap:
                continue
            arr = (_TF @ np.asarray(corners, np.float64).T).T  # (#corners,3) -> world frame
            for c in range(len(corners)):
                cands = cmap.get(perm[c])      # proc corner c is canonical corner perm[c]
                if not cands:
                    continue
                o = arr[c]
                # assign to the instance whose LBS prediction is nearest (duplicate-ID swap)
                idx = min(cands, key=lambda ci: ((lbsw[ci[1]] - o) ** 2).sum())[1]
                res = o - lbsw[idx]
                if drop_thresh and np.linalg.norm(res) > drop_thresh:
                    continue                                   # reject gross outlier
                residual_m[i, idx] = res
                mask[i, idx] = 1
        pose6d[i] = rotmat_to_6d(rot)
        root_pos[i] = root_all[bf]
        bvh_idx[i] = bf

    residual_scaled = (residual_m * SCALE_FACTOR).astype(np.float32)
    os.makedirs(f"{out_dir}/{S}", exist_ok=True)
    np.savez_compressed(
        out_npz,
        residual_m=residual_m.astype(np.float16),
        residual_scaled=residual_scaled.astype(np.float16),
        mask=mask,
        pose6d=pose6d, root_position=root_pos, j_rest=j_rest.astype(np.float32),
        parents=parents.astype(np.int32), frame_index=bvh_idx,
        marker_ids=np.array(markers), scale_factor=np.float32(SCALE_FACTOR),
        global_R=gR.astype(np.float32), global_t=gt.astype(np.float32))
    obs = residual_m[mask.astype(bool)]
    log(f"  {S}/{sh}: F={F} (bvh {Nb}, proc {Np}) M={M} obs={int(mask.sum())} "
        f"resid={np.linalg.norm(obs,axis=1).mean()*100:.2f}cm -> {os.path.getsize(out_npz)/1e6:.1f}MB")
    del proc, residual_m, residual_scaled, mask, pose6d; gc.collect()
    return "ok"


def write_canonical(S, model, out_dir):
    os.makedirs(f"{out_dir}/{S}", exist_ok=True)
    np.savez_compressed(f"{out_dir}/{S}/canonical.npz",
                        marker_ids=np.array(model["markers"]),
                        p_canon=model["p_canon"].astype(np.float32),
                        p_bind=model["p_bind"].astype(np.float32),
                        W=model["W"].astype(np.float32),
                        joint_names=np.array(model["joint_names"]))


def validate_s1(out_dir, log):
    """Compare regenerated S1/shot_001 residual to the existing Blender residual file."""
    npz = f"{out_dir}/S1/shot_001.npz"
    if not os.path.exists(npz):
        log("  validate: S1/shot_001.npz not found"); return
    d = np.load(npz, allow_pickle=True)
    rs = d["residual_scaled"].astype(np.float32); mids = list(d["marker_ids"]); fidx = d["frame_index"]
    midx = {m: i for i, m in enumerate(mids)}
    exist = json.load(open(f"{STATIC}/S1/raw/shot_001_captury/"
                           "S1_residuals_shot_001_world_lbs_scaled_tpose.json"))
    diffs = []
    for i in range(0, len(fidx), 200):                # sample
        ek = str(int(fidx[i]))                         # existing keyed by bvh frame
        if ek not in exist: continue
        for m, v in exist[ek].items():
            if m in midx and np.any(rs[i, midx[m]]):
                diffs.append(np.linalg.norm(rs[i, midx[m]] - np.array(v)))
    diffs = np.array(diffs)
    log(f"  validate S1/shot_001 vs existing: mean={diffs.mean()/SCALE_FACTOR*100:.2f}cm "
        f"(expected ~6cm marker-vs-skin LBS method gap), n={len(diffs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--subjects", nargs="+", default=list(SUBJECTS))
    ap.add_argument("--validate", action="store_true", help="after S1, compare to existing residuals")
    ap.add_argument("--drop", type=float, default=0.20,
                    help="reject observed markers >this many metres from LBS (0.2 matches old pipeline)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    logf = open(f"{args.out}/run.log", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    t0 = time.time()
    counts = {}
    for S in args.subjects:
        log(f"=== {S} ({SUBJECTS[S]}) ===")
        model = load_subject_model(S)
        write_canonical(S, model, args.out)
        for sh in shots_of(S):
            try:
                r = process_shot(S, sh, model, args.out, log, drop_thresh=args.drop)
            except Exception as e:
                r = "error"; log(f"  {S}/{sh}: ERROR {type(e).__name__}: {e}")
            counts[r] = counts.get(r, 0) + 1
        del model; gc.collect()
        if S == "S1" and args.validate:
            validate_s1(args.out, log)
    log(f"DONE in {(time.time()-t0)/60:.1f} min  counts={counts}")


if __name__ == "__main__":
    main()
