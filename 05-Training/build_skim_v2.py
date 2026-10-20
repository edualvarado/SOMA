"""
build_skim_v2.py -- derive SKIM v2 from v1 with the retraining fixes baked into the data (README.md, SKIM v2 vs v1).

v1 defines   observed = global_R @ LBS_v1 + global_t + residual_v1   (world Z-up, metres), with
LBS_v1 = LBS(p_bind, W_v1, pose_bvh, j_rest_v1, root). v2 keeps the observed markers, global_R/t and masks, and
re-references the residual to the rebuilt rig and the de-twisted pose:
    residual_v2 = observed - (global_R_v2 @ LBS_v2 + global_t_v2),   LBS_v2 = LBS(p_bind, W_rig, pose_v2, j_rest_rig, root_v2)
with global_R/t refitted per shot, so the v1 identity holds with the v2 keys. Changes per subject:
  * rig (05-Training/rigs/<S>_rig.npz, shipped as <S>/canonical_model/<S>_rig.npz): rest joints (includes the S2-S5 Hips rest offset) + skin weights (no
    zero-weight hand vertices); marker weights = barycentric blend of the skin weights (as 01 computes them).
  * root trajectory: the BVH root channel tracks Captury's hip point (S2-S5: ~6 cm from the rig's Hips); root_position
    is re-expressed at the rig Hips with a per-shot offset c fitted to the observations (root_position_raw kept).
  * rest-skeleton convention: S3 shot_005/008 come from a BVH export whose rest skeleton faces the other way (180 deg
    about Y); their rotations are re-expressed on the rig skeleton (pose_rest_rotation).
  * forearm de-twist (S2-S5): ForeArm twist removed from the pose; the raw pose is kept as pose6d_raw / pose_raw.
  * S2-S5 barycentric maps replaced by the fixed maps (_bary_fix_2026-09-24); S1 keeps its shipped map.
  * S4 inner layers (muscle/bone layers and meshes, T- and A-pose, per-muscle meshes) moved by LAYER_REST_OFFSET.
  * packed training files rebuilt with global_R (all subjects).
Everything else is copied unchanged.

Run (from 05-Training/):
  python build_skim_v2.py --src <v1 dataset dir> --bary-fix <_bary_fix_2026-09-24 dir> --out <v2 dir>
"""
import os, sys, glob, json, shutil, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forearm_detwist import detwist

DETWIST_SUBJECTS = {"S2", "S3", "S4", "S5"}               # 01 --forearm-detwist auto (off for S1)
BARY_FIX_SUBJECTS = {"S2", "S3", "S4", "S5"}              # S1 keeps its shipped map
ROOT_FIT_SUBJECTS = {"S1", "S2", "S3", "S4", "S5"}        # per-shot root offset (BVH root point vs rig Hips)
LAYER_REST_OFFSET = {"S4": (0.002098, 0.014745, 0.010864)}  # as 01/02/03 (m, mesh frame)
INNER_MESHES = ["musc_layer", "musc_mesh", "skel_layer", "skel_mesh"]
ROT_X90 = torch.tensor([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]], dtype=torch.float64)  # BVH (Y-up) -> world Z-up
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def lbs(P, W, R, j_rest, parents, root):
    """Batched marker LBS, same math as 01's lbs_working_batch_rotmat / _skim_lbs.lbs_frame (BVH frame).
    P (M,3), W (M,J), R (B,J,3,3), j_rest (J,3), root (B,3) -> (B,M,3)."""
    B, J = R.shape[0], len(parents)
    G, T = [None] * J, [None] * J
    for i in range(J):
        p = int(parents[i])
        t = root if p < 0 else (j_rest[i] - j_rest[p]).expand(B, 3)
        if p < 0:
            G[i], T[i] = R[:, i], t
        else:
            G[i], T[i] = G[p] @ R[:, i], T[p] + (G[p] @ t.unsqueeze(-1)).squeeze(-1)
    G, T = torch.stack(G, 1), torch.stack(T, 1)                      # posed joint frames (B,J,3,3), (B,J,3)
    T = T - (G @ j_rest.unsqueeze(-1)).squeeze(-1)                   # skinning: G (x - j_rest) + T
    return torch.einsum('mj,bjxy,my->bmx', W, G, P) + torch.einsum('mj,bjx->bmx', W, T)


def sixd_rows_to_R(p):
    """shot npz pose6d (R[:, :2] row-major) -> R."""
    p = p.reshape(p.shape[:-1] + (3, 2))
    return gram_schmidt(p[..., 0], p[..., 1])


def sixd_cols_to_R(p):
    """packed pose ([col0|col1]) -> R."""
    return gram_schmidt(p[..., :3], p[..., 3:])


def gram_schmidt(a, b):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b - (a * b).sum(-1, keepdim=True) * a
    b = b / b.norm(dim=-1, keepdim=True)
    return torch.stack([a, b, torch.cross(a, b, dim=-1)], -1)


def rest_rotation(j_shot, j_ref, parents):
    """Rotation M with the shot's rest bone offsets = M @ the reference (rig BVH) offsets: I, or 180 deg about Y for BVH
    exports whose rest skeleton faces the other way (S3 shot_005/008). Raises if neither fits."""
    rel = lambda j: np.stack([j[i] - j[p] for i, p in enumerate(parents) if p >= 0])
    for M in (np.eye(3), np.diag([-1.0, 1.0, -1.0])):
        if np.abs(rel(j_shot) - rel(j_ref) @ M.T).max() < 2e-3:
            return M
    raise ValueError("shot rest skeleton matches the rig skeleton neither as-is nor rotated 180 deg about Y")


def to_rig_frame(R, M):
    """Local rotations (B,J,3,3) posed on a rest skeleton rotated by M -> the same pose on the reference skeleton:
    root R0 M, children M^T R M (global frames G M), so every posed bone G (M o) = (G M) o."""
    if np.allclose(M, np.eye(3)):
        return R
    M = torch.as_tensor(M, dtype=R.dtype, device=R.device)
    out = M.T @ R @ M
    out[:, 0] = R[:, 0] @ M
    return out


def marker_weights(bary_map, marker_ids, skin_W):
    W = np.zeros((len(marker_ids), skin_W.shape[1]))
    for k, m in enumerate(marker_ids):
        e = bary_map[str(m)]
        W[k] = (np.ravel(e["bary_coords"])[:3, None] * skin_W[e["vertex_indices"]]).sum(0)
    return W


def shift_obj(src, dst, t):
    """Add t to every 'v x y z' line (6 decimals, as the Blender exports); other lines byte-identical."""
    out, n = [], 0
    with open(src, newline="") as f:
        for line in f:
            if line.startswith("v "):
                body = line.rstrip("\r\n"); eol = line[len(body):]
                x = np.array(body.split()[1:4], float) + t
                line = f"v {x[0]:.6f} {x[1]:.6f} {x[2]:.6f}{eol}"; n += 1
            out.append(line)
    with open(dst, "w", newline="") as f:
        f.writelines(out)
    return n


def weights_json(entries, W, names, is_skin, marker_ids=None):
    """Rewrite bone_names/weights (and bone_indices = 24-joint BVH index) from the dense matrix W.
    Skin: list ordered by vertex. Markers: dict; markers not in marker_ids (unused) keep their original Blender entry."""
    def sparse(w):
        j = [i for i in np.argsort(-w) if w[i] > 0]
        return {"bone_indices": [int(i) for i in j], "weights": [float(w[i]) for i in j], "bone_names": [names[i] for i in j]}
    if is_skin:
        for v, e in enumerate(entries):
            e.update(sparse(W[v]))
        return entries
    row = {str(m): k for k, m in enumerate(marker_ids)}
    for m, e in entries.items():
        if m in row:
            e.update(sparse(W[row[m]]))
    return entries


def build_subject(S, src, out, bary_fix, rigs, log):
    s_src, s_out = os.path.join(src, S), os.path.join(out, S)
    shutil.copytree(s_src, s_out, ignore=shutil.ignore_patterns("shot_*.npz"), dirs_exist_ok=True)
    cm = os.path.join(s_out, "canonical_model")

    # --- barycentric map ---
    if S in BARY_FIX_SUBJECTS:
        shutil.copy2(os.path.join(bary_fix, f"{S}_generated_marker_barycentric_map.json"),
                     os.path.join(cm, "generated_marker_barycentric_map.json"))
    bary = json.load(open(os.path.join(cm, "generated_marker_barycentric_map.json")))

    # --- rig: skin weights, marker weights, rest joints ---
    rig = np.load(os.path.join(rigs, f"{S}_rig.npz"))
    names = [str(n) for n in rig["joint_names"]]
    skin_W = rig["weights"].astype(np.float64)
    canon = dict(np.load(os.path.join(s_src, "canonical.npz")))
    if [str(n) for n in canon["joint_names"]] != names:
        raise ValueError(f"{S}: rig joint order differs from canonical.npz")
    if sorted(bary) != [str(m) for m in canon["marker_ids"]]:
        raise ValueError(f"{S}: barycentric map keys differ from canonical.npz marker_ids")
    W_new = marker_weights(bary, canon["marker_ids"], skin_W)
    j_new = rig["j_rest"].astype(np.float64)

    shutil.copy2(os.path.join(rigs, f"{S}_rig.npz"), os.path.join(cm, f"{S}_rig.npz"))  # what 01/02/03 load on v2
    npy = os.path.join(cm, "lbs_skin", f"{S}_skin_lbs_weights_exported.npy")
    old_npy = np.load(npy)
    if old_npy.shape != skin_W.shape:
        raise ValueError(f"{S}: skin weight shape {old_npy.shape} vs rig {skin_W.shape}")
    np.save(npy, rig["weights"].astype(old_npy.dtype))
    sj = os.path.join(cm, "lbs_skin", f"{S}_skin_lbs_weights_exported.json")
    json.dump(weights_json(json.load(open(sj)), skin_W, names, True), open(sj, "w"))
    mj = os.path.join(cm, "lbs_markers", f"{S}_marker_lbs_weights_exported.json")
    json.dump(weights_json(json.load(open(mj)), W_new, names, False, [str(m) for m in canon["marker_ids"]]), open(mj, "w"))

    # --- S4 inner layers ---
    if S in LAYER_REST_OFFSET:
        t = np.asarray(LAYER_REST_OFFSET[S])
        objs = [os.path.join(s_out, "layers", pose, f"{m}-{S}-{P}.obj") for pose, P in (("tpose", "TPose"), ("apose", "APose")) for m in INNER_MESHES]
        objs += sorted(glob.glob(os.path.join(s_out, "layers", "tpose", "muscle_meshes_tpose", "*.obj")))
        for p in objs:
            rel = os.path.relpath(p, s_out)
            shift_obj(os.path.join(s_src, rel), p, t)
        log(f"  {S}: shifted {len(objs)} inner-layer OBJs by {t * 1000} mm")

    # --- residuals and poses ---
    P = torch.from_numpy(canon["p_bind"].astype(np.float64)).to(DEV)
    W_old_t, W_new_t = torch.from_numpy(canon["W"].astype(np.float64)).to(DEV), torch.from_numpy(W_new).to(DEV)
    j_new_t = torch.from_numpy(j_new).to(DEV)
    rot = ROT_X90.to(DEV)
    shots = sorted(glob.glob(os.path.join(s_src, "shot_*.npz")))

    j_ref = rig["j_rest_captury"].astype(np.float64)

    def frames(d, pk, sl, Mrest):
        """Observed markers (v1 identity), v2 marker LBS with the raw root (world Z-up), the rig-frame root rotation
        in world, and the v2 poses, for the frames in sl."""
        parents = d["parents"].tolist()
        R = sixd_rows_to_R(torch.from_numpy(d["pose6d"][sl].astype(np.float64)).to(DEV))         # (B,J,3,3) BVH local
        Rc = to_rig_frame(R, Mrest)                                                               # on the rig skeleton
        Rdt = detwist(Rc, j_new_t, names, parents) if S in DETWIST_SUBJECTS else Rc
        root = torch.from_numpy(d["root_position"][sl].astype(np.float64)).to(DEV)
        j1 = torch.from_numpy(d["j_rest"].astype(np.float64)).to(DEV)
        gR1, gt1 = (torch.from_numpy(d[k].astype(np.float64)).to(DEV) for k in ("global_R", "global_t"))
        obs = (lbs(P, W_old_t, R, j1, parents, root) @ rot.T) @ gR1.T + gt1 \
            + torch.from_numpy(d["residual_m"][sl].astype(np.float64)).to(DEV)
        L2 = lbs(P, W_new_t, Rdt, j_new_t, parents, root) @ rot.T
        return obs, L2, rot @ Rc[:, 0], Rc, Rdt, root

    def fit_alignment(d, pk, Mrest):
        """Per-shot rigid alignment (gR, gt) of the v2 LBS to the observations and root offset c (rig Hips -> BVH root
        point, root frame), alternating least squares on every 5th frame. The LBS with root offset c is L2 - A c."""
        sl = slice(0, d["residual_m"].shape[0], 5)
        obs, L2, A, _, _, _ = frames(d, pk, sl, Mrest)
        m = torch.from_numpy(d["mask"][sl].astype(bool)).to(DEV)
        c = torch.zeros(3, dtype=torch.float64, device=DEV)
        for _ in range(10):
            L = L2 - (A @ c).unsqueeze(1)
            X, Y = L[m], obs[m]
            mx, my = X.mean(0), Y.mean(0)
            U, _, Vt = torch.linalg.svd((Y - my).T @ (X - mx))
            D = torch.eye(3, dtype=torch.float64, device=DEV); D[2, 2] = torch.sign(torch.linalg.det(U @ Vt))
            gR = U @ D @ Vt; gt = my - gR @ mx
            if S not in ROOT_FIT_SUBJECTS:
                break
            r = obs - L2 @ gR.T - gt                                                             # r(c) = r + gR A c
            B = (gR @ A).unsqueeze(1).expand(-1, L2.shape[1], -1, -1)[m]                         # (N,3,3)
            c = -torch.linalg.solve((B.transpose(-1, -2) @ B).sum(0), (B.transpose(-1, -2) @ r[m].unsqueeze(-1)).sum(0).squeeze(-1))
        return gR, gt, c

    def shot_v2(d, pk, gR, gt, c, Mrest):
        """v2 residual (F,M,3) m (observed - (gR LBS_v2 + gt)), root trajectory, shot-npz and packed poses."""
        F = d["residual_m"].shape[0]; parents = d["parents"].tolist()
        r2 = np.zeros((F, P.shape[0], 3)); root2 = np.zeros((F, 3))
        pose_dt = np.zeros(d["pose6d"].shape, np.float32); packed_dt = np.zeros(pk["pose"].shape, np.float32)
        for k in range(0, F, 500):
            sl = slice(k, min(k + 500, F)); B = sl.stop - sl.start
            obs, L2, A, Rc, Rdt, root = frames(d, pk, sl, Mrest)
            m = torch.from_numpy(d["mask"][sl].astype(np.float64)).to(DEV)[..., None]
            r2[sl] = ((obs - (L2 - (A @ c).unsqueeze(1)) @ gR.T - gt) * m).cpu().numpy()
            root2[sl] = (root - Rc[:, 0] @ c).cpu().numpy()                                     # rig Hips trajectory
            pose_dt[sl] = Rdt[..., :, :2].reshape(B, -1, 6).cpu().numpy()
            Rp = to_rig_frame(sixd_cols_to_R(torch.from_numpy(pk["pose"][sl].astype(np.float64)).to(DEV).view(B, -1, 6)), Mrest)
            Rp_dt = detwist(Rp, j_new_t, names, parents) if S in DETWIST_SUBJECTS else Rp
            packed_dt[sl] = torch.cat([Rp_dt[..., :, 0], Rp_dt[..., :, 1]], -1).reshape(B, -1).cpu().numpy()
        if S not in DETWIST_SUBJECTS and np.allclose(Mrest, np.eye(3)):  # identical poses: keep v1's bytes
            pose_dt, packed_dt = d["pose6d"].astype(np.float32), pk["pose"].astype(np.float32)
        return r2, root2, pose_dt, packed_dt

    def load(shot_p):
        sh = os.path.basename(shot_p)[:-4]
        d = dict(np.load(shot_p, allow_pickle=True))
        pk = dict(np.load(os.path.join(s_src, "preprocessed_vFinal_clean", f"{sh}.npz")))
        if pk["pose"].shape[0] != d["residual_m"].shape[0] or not np.array_equal(pk["masks"], d["mask"]):
            raise ValueError(f"{S}/{sh}: packed and shot npz disagree")
        return sh, d, pk

    # Per shot: (1) rest-skeleton convention of the BVH export (S3 shot_005/008 face the other way); (2) root offset: the
    # BVH root channel tracks Captury's hip point, not the rig's Hips (S2-S5 tracking backups: zero Hips rest offset,
    # ~6 cm; S3 shot_005/008: main export; S1: re-exported BVHs with a Hips rest offset that varies between shots);
    # (3) the LBS -> observed rigid alignment global_R/global_t, refitted to the v2 LBS (v1's was fitted to the v1 LBS).
    resid_cm = []
    for shot_p in shots:
        sh, d, pk = load(shot_p)
        F, parents = d["residual_m"].shape[0], d["parents"].tolist()
        Mrest = rest_rotation(d["j_rest"].astype(np.float64), j_ref, parents)
        gR, gt, c = fit_alignment(d, pk, Mrest)
        r2, root2, pose_dt, packed_dt = shot_v2(d, pk, gR, gt, c, Mrest)
        gR, gt, c = (x.cpu().numpy() for x in (gR, gt, c))
        new = dict(d)
        new.update(residual_m=r2.astype(np.float16), residual_scaled=(r2 * 0.1).astype(np.float16),
                   pose6d=pose_dt, pose6d_raw=d["pose6d"].astype(np.float32), j_rest=j_new.astype(np.float32),
                   root_position=root2.astype(np.float32), root_position_raw=d["root_position"].astype(np.float32),
                   root_offset=c.astype(np.float32), pose_rest_rotation=Mrest.astype(np.float32),
                   global_R=gR.astype(np.float32), global_t=gt.astype(np.float32))
        np.savez_compressed(os.path.join(s_out, f"{sh}.npz"), **new)
        np.savez_compressed(os.path.join(s_out, "preprocessed_vFinal_clean", f"{sh}.npz"),
                            pose=packed_dt, pose_raw=pk["pose"].astype(np.float32), residuals=(r2 * 0.1).astype(np.float16),
                            masks=d["mask"].astype(np.uint8), global_R=gR.astype(np.float32))
        mk = d["mask"].astype(bool)
        n1, n2 = np.linalg.norm(d["residual_m"][mk].astype(np.float64), axis=1), np.linalg.norm(r2[mk], axis=1)
        resid_cm.append(n2.mean() * 100)
        dR = np.degrees(np.arccos(np.clip((np.trace(gR.T @ d["global_R"]) - 1) / 2, -1, 1)))
        log(f"  {S}/{sh}: F={F}{' REST ROTATED 180' if not np.allclose(Mrest, np.eye(3)) else ''} root offset {np.round(c * 1000, 1)} mm, "
            f"alignment change {dR:.2f} deg {np.linalg.norm(gt - d['global_t']) * 1000:.0f} mm | mean |r| v1 {n1.mean() * 1000:.1f} -> v2 {n2.mean() * 1000:.1f} mm")

    canon.update(W=W_new.astype(np.float32), j_rest=j_new.astype(np.float32), parents=np.asarray(parents, np.int32))
    np.savez_compressed(os.path.join(s_out, "canonical.npz"), **canon)
    return float(np.mean(resid_cm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="v1 dataset dir (with <S>/shot_*.npz)")
    ap.add_argument("--bary-fix", required=True, help="dir with <S>_generated_marker_barycentric_map.json")
    ap.add_argument("--rigs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "rigs"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", nargs="+", default=["S1", "S2", "S3", "S4", "S5"])
    a = ap.parse_args()
    if os.path.exists(a.out) and os.listdir(a.out):
        ap.error(f"{a.out} is not empty")
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "..", f"build_{os.path.basename(os.path.normpath(a.out))}.log"), "a")
    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()
    log(f"build_skim_v2: src={a.src} out={a.out} device={DEV}")
    for f in os.listdir(a.src):
        if os.path.isfile(os.path.join(a.src, f)):
            shutil.copy2(os.path.join(a.src, f), os.path.join(a.out, f))
    meta = json.load(open(os.path.join(a.src, "metadata.json")))
    for S in a.subjects:
        meta["subjects"][S]["mean_residual_cm"] = round(build_subject(S, a.src, a.out, a.bary_fix, a.rigs, log), 2)
    meta["dataset_version"] = 2
    meta["per_shot_npz_keys"].update({
        "pose6d": "(F,J,6) float32, per-joint 6D rotation (first two columns of the rotation matrix); v2: ForeArm twist removed (S2-S5)",
        "pose6d_raw": "(F,J,6) float32, pose6d as in the BVH (ForeArm twist kept)",
        "j_rest": "(J,3) float32, rest joint positions of the rebuilt rig (metres, BVH frame)",
        "root_position": "(F,3) float32, rig Hips world position (metres, BVH frame) = root_position_raw - R_root @ root_offset",
        "root_position_raw": "(F,3) float32, BVH root channel (metres)",
        "root_offset": "(3,) float32, BVH root point relative to the rig Hips in the root frame (metres), fitted to the shot's observations",
        "global_R": "(3,3) float32 per-shot rigid alignment LBS->observed (v2: refitted to the v2 LBS)",
        "global_t": "(3,) float32 per-shot translation (v2: refitted to the v2 LBS)",
        "pose_rest_rotation": "(3,3) float32, M with the BVH rest skeleton = M @ rig skeleton (identity, or 180 deg about Y for S3 shot_005/008); pose6d = pose6d_raw re-expressed on the rig (root R0 M, others M^T R M)",
        "residual_m": "(F,M,3) float16, observed - (global_R @ LBS + global_t) in metres, LBS = LBS(p_bind, W, pose6d, j_rest, root_position) (0 where unobserved)"})
    meta["canonical_npz_keys"].update({
        "W": "(M,J) marker LBS weights: barycentric blend of the rig skin weights",
        "j_rest": "(J,3) rest joint positions of the rebuilt rig (metres, BVH frame)",
        "parents": "(J,) int32 joint parents (root=-1)"})
    meta["preprocessed_npz_keys"].update({
        "pose": "(F,J*6) float32, training 6D pose (root rotated to Z-up; column order [col0|col1]); ForeArm twist removed (S2-S5)",
        "pose_raw": "(F,J*6) float32, pose with the BVH ForeArm twist",
        "global_R": "(3,3) float32, per-shot LBS->observed alignment; loaders rotate the residuals by global_R^T"})
    meta["canonical_model"].update({
        "lbs_markers/<S>_marker_lbs_weights_exported.json": "per-marker {bone_indices, bone_names, weights}; v2: the M used markers carry the rig weights (= canonical.npz W, bone_indices = 24-joint index), the unused ones keep their original 56-bone Blender entry",
        "lbs_skin/<S>_skin_lbs_weights_exported.json": "per-skin-vertex {blender_v_idx, coords_local, bone_indices, bone_names, weights}; v2: rig weights on the 24-joint skeleton (bone_indices = 24-joint index), no zero rows",
        "lbs_skin/<S>_skin_lbs_weights_exported.npy": "(23752,J) float32, the same rig weights as the JSON"})
    meta["canonical_model"]["<S>_rig.npz"] = ("SOMA rig (build_rig.py): j_rest (J,3) rest joints refitted to the bones, weights (23752,J) skin "
                                              "weights, joint_names, j_rest_captury (BVH rest joints + Hips offset, reference only)")
    meta["raw_bvh"]["root"] = "BVH root channel = shot npz root_position_raw; S3 shot_005/008 come from a different export (rest skeleton rotated 180 deg about Y, see pose_rest_rotation)"
    json.dump(meta, open(os.path.join(a.out, "metadata.json"), "w"), indent=2)
    log("done")


if __name__ == "__main__":
    main()
