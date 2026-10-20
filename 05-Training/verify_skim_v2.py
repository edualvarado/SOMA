"""
verify_skim_v2.py -- check a SKIM V2 build (build_skim_v2.py) against its V1 source.

  1. files: everything outside the intended changes is byte-identical to V1
  2. observed markers: global_R @ LBS + global_t + residual_m is the same with the V1 and the V2 keys (every frame)
  3. packed files: residuals == residual_scaled, masks, global_R, pose_raw == V1 pose, pose == pose6d (non-root joints)
  4. weights: no zero rows; the skin JSON parses (by bone name) to the npy; marker W = barycentric blend of the npy
  5. S4 inner meshes: moved onto the skin (shared head vertices), non-vertex lines unchanged
  6. poses: the V2 pose on the rig skeleton gives the same posed skeleton as the raw pose on the shot's own BVH skeleton
     (up to the rig's refitted bone lengths), i.e. no rest-convention mix-up
  7. training targets: V2 packed target vs 01's V1 conversion, per-frame centred (informational: the conversion biases)

Run (from 05-Training/):  python verify_skim_v2.py --v1 <V1 dataset dir> --v2 <V2 dir>
"""
import os, sys, glob, json, hashlib, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_skim_v2 import (lbs, sixd_rows_to_R, sixd_cols_to_R, ROT_X90, DEV, LAYER_REST_OFFSET, INNER_MESHES,
                           DETWIST_SUBJECTS, BARY_FIX_SUBJECTS)
from forearm_detwist import detwist
from scipy.spatial.transform import Rotation

R2M = torch.from_numpy((Rotation.from_euler('y', 90, degrees=True) * Rotation.from_euler('x', -90, degrees=True)).as_matrix())
ok = True


def check(cond, msg):
    global ok
    ok &= bool(cond)
    print(("  OK   " if cond else "  FAIL ") + msg, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def expected_changes(S):
    ch = {f"{S}/canonical.npz", f"{S}/canonical_model/lbs_skin/{S}_skin_lbs_weights_exported.npy",
          f"{S}/canonical_model/lbs_skin/{S}_skin_lbs_weights_exported.json",
          f"{S}/canonical_model/lbs_markers/{S}_marker_lbs_weights_exported.json"}
    if S in BARY_FIX_SUBJECTS:
        ch.add(f"{S}/canonical_model/generated_marker_barycentric_map.json")
    if S in LAYER_REST_OFFSET:
        ch |= {f"{S}/layers/{p}/{m}-{S}-{P}.obj" for p, P in (("tpose", "TPose"), ("apose", "APose")) for m in INNER_MESHES}
    return ch


def fk(R, j, parents):
    """Posed joint positions (B,J,3), root at 0."""
    G, T = [None] * len(parents), [None] * len(parents)
    for i, p in enumerate(parents):
        if p < 0:
            G[i], T[i] = R[:, i], torch.zeros(R.shape[0], 3, dtype=R.dtype, device=R.device)
        else:
            G[i] = G[p] @ R[:, i]; T[i] = T[p] + (G[p] @ (j[i] - j[p]))
    return torch.stack(T, 1)


def obj_v(p):
    return np.array([l.split()[1:4] for l in open(p) if l.startswith("v ")], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True); ap.add_argument("--v2", required=True)
    ap.add_argument("--rigs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "rigs"))
    ap.add_argument("--subjects", nargs="+", default=["S1", "S2", "S3", "S4", "S5"])
    a = ap.parse_args()
    v1, v2 = a.v1, a.v2

    print("[1] files")
    f1 = {os.path.relpath(p, v1).replace("\\", "/") for p in glob.glob(f"{v1}/**/*", recursive=True) if os.path.isfile(p)}
    f2 = {os.path.relpath(p, v2).replace("\\", "/") for p in glob.glob(f"{v2}/**/*", recursive=True) if os.path.isfile(p)}
    added = {f"{S}/canonical_model/{S}_rig.npz" for S in a.subjects}
    check(f1 | added == f2, f"same file set + the rigs ({len(f2)} files){'' if f1 | added == f2 else f': only V1 {sorted(f1 - f2)[:5]}, only V2 {sorted(f2 - f1 - added)[:5]}'}")
    changed = {"metadata.json"}
    for S in a.subjects:
        changed |= expected_changes(S)
        changed |= {os.path.relpath(p, v2).replace("\\", "/") for p in glob.glob(f"{v2}/{S}/shot_*.npz")}
        changed |= {os.path.relpath(p, v2).replace("\\", "/") for p in glob.glob(f"{v2}/{S}/preprocessed_vFinal_clean/*.npz")}
        if S in LAYER_REST_OFFSET:
            changed |= {os.path.relpath(p, v2).replace("\\", "/") for p in glob.glob(f"{v2}/{S}/layers/tpose/muscle_meshes_tpose/*.obj")}
    same = sorted((f1 & f2) - changed)
    diff = [p for p in same if os.path.getsize(f"{v1}/{p}") != os.path.getsize(f"{v2}/{p}") or sha(f"{v1}/{p}") != sha(f"{v2}/{p}")]
    check(not diff, f"{len(same)} unchanged files byte-identical{'' if not diff else f': differ {diff[:5]}'}")
    nochange = [p for p in sorted(changed & f1) if sha(f"{v1}/{p}") == sha(f"{v2}/{p}")]
    check(not nochange, f"{len(changed & f1)} intended changes all differ from V1{'' if not nochange else f': unchanged {nochange[:5]}'}")
    meta = json.load(open(f"{v2}/metadata.json"))
    check(meta.get("dataset_version") == 2, "metadata.json dataset_version == 2")

    rot = ROT_X90.to(DEV)
    for S in a.subjects:
        print(f"[{S}]")
        c1, c2 = np.load(f"{v1}/{S}/canonical.npz"), np.load(f"{v2}/{S}/canonical.npz")
        rig = np.load(os.path.join(a.rigs, f"{S}_rig.npz")); names = [str(n) for n in rig["joint_names"]]
        P = torch.from_numpy(c1["p_bind"].astype(np.float64)).to(DEV)
        check(np.array_equal(c1["p_bind"], c2["p_bind"]) and np.array_equal(c1["marker_ids"], c2["marker_ids"]), "p_bind / marker_ids unchanged")

        # [4] weights
        npy = np.load(f"{v2}/{S}/canonical_model/lbs_skin/{S}_skin_lbs_weights_exported.npy")
        check(np.array_equal(npy, rig["weights"].astype(npy.dtype)), "skin npy == rig weights")
        check((npy.sum(1) > 0.99).all() and (c2["W"].sum(1) > 0.99).all(), f"no zero-weight rows (skin min {npy.sum(1).min():.4f}, markers min {c2['W'].sum(1).min():.4f})")
        js = json.load(open(f"{v2}/{S}/canonical_model/lbs_skin/{S}_skin_lbs_weights_exported.json"))
        Wj = np.zeros_like(npy); jx = {n: i for i, n in enumerate(names)}
        for v, e in enumerate(js):
            for n, w in zip(e["bone_names"], e["weights"]):
                Wj[v, jx[n]] = w
        check(np.abs(Wj - npy).max() < 1e-6, f"skin JSON == npy (max {np.abs(Wj - npy).max():.1e})")
        bary = json.load(open(f"{v2}/{S}/canonical_model/generated_marker_barycentric_map.json"))
        Wm = np.stack([(np.ravel(bary[str(m)]["bary_coords"])[:3, None] * npy[bary[str(m)]["vertex_indices"]]).sum(0) for m in c2["marker_ids"]])
        check(np.abs(Wm - c2["W"]).max() < 1e-5, f"canonical W == barycentric blend of the skin weights (max {np.abs(Wm - c2['W']).max():.1e})")
        bw = np.array([np.ravel(e["bary_coords"])[:3] for e in bary.values()])
        check(bw.min() > -0.1 and bw.max() < 1.01, f"barycentric weights in range [{bw.min():.3f}, {bw.max():.3f}]")
        check(np.array_equal(c2["j_rest"], rig["j_rest"]), "canonical j_rest == rig j_rest")
        shipped = np.load(f"{v2}/{S}/canonical_model/{S}_rig.npz")
        check(all(np.array_equal(shipped[k], rig[k]) for k in rig.files), "shipped canonical_model rig == rigs/ rig")

        # [5] S4 meshes
        if S in LAYER_REST_OFFSET:
            t = np.asarray(LAYER_REST_OFFSET[S])
            for pose, PP in (("tpose", "TPose"), ("apose", "APose")):
                sk = obj_v(f"{v2}/{S}/layers/{pose}/skin_layer-{S}-{PP}.obj")
                mu1, mu2 = obj_v(f"{v1}/{S}/layers/{pose}/musc_layer-{S}-{PP}.obj"), obj_v(f"{v2}/{S}/layers/{pose}/musc_layer-{S}-{PP}.obj")
                check(np.abs(mu2 - mu1 - t).max() < 1.5e-6, f"{pose} musc_layer moved by t (max err {np.abs(mu2 - mu1 - t).max():.1e})")
                n_shared = int((np.abs(sk - mu2).max(1) < 2e-6).sum())
                check(n_shared > 6000, f"{pose} musc_layer shares {n_shared} vertices with the skin (V1: {int((np.abs(sk - mu1).max(1) < 2e-6).sum())})")
            for p in glob.glob(f"{v2}/{S}/layers/*/*.obj") + glob.glob(f"{v2}/{S}/layers/tpose/muscle_meshes_tpose/*.obj"):
                q = p.replace(v2, v1)
                l1 = [l for l in open(q) if not l.startswith("v ")]; l2 = [l for l in open(p) if not l.startswith("v ")]
                if l1 != l2:
                    check(False, f"non-vertex lines changed in {os.path.relpath(p, v2)}")

        # [2] observed markers, [3] packed, [6] targets
        W1, W2 = [torch.from_numpy(c["W"].astype(np.float64)).to(DEV) for c in (c1, c2)]
        jc = torch.from_numpy(rig["j_rest_captury"].astype(np.float64)).to(DEV)
        e_obs, e_pk, e_tgt, n_obs, e_root, e_fk = 0.0, 0.0, [], 0, 0.0, 0.0
        for sp in sorted(glob.glob(f"{v2}/{S}/shot_*.npz")):
            sh = os.path.basename(sp)[:-4]
            d1, d2 = np.load(f"{v1}/{S}/{sh}.npz", allow_pickle=True), np.load(sp, allow_pickle=True)
            p1, p2 = np.load(f"{v1}/{S}/preprocessed_vFinal_clean/{sh}.npz"), np.load(f"{v2}/{S}/preprocessed_vFinal_clean/{sh}.npz")
            same_keys = [k for k in ("mask", "parents", "frame_index", "marker_ids", "scale_factor") if not np.array_equal(d1[k], d2[k])]
            if same_keys:
                check(False, f"{sh}: keys changed {same_keys}")
            if not (np.array_equal(p2["residuals"], d2["residual_scaled"]) and np.array_equal(p2["masks"], d2["mask"])
                    and np.array_equal(p2["global_R"], d2["global_R"]) and np.array_equal(p2["pose_raw"], p1["pose"])
                    and np.array_equal(d2["pose6d_raw"], d1["pose6d"]) and np.array_equal(d2["root_position_raw"], d1["root_position"])):
                check(False, f"{sh}: packed/shot npz inconsistent")
            F = d1["mask"].shape[0]; parents = d1["parents"].tolist()
            j1, j2 = [torch.from_numpy(d["j_rest"].astype(np.float64)).to(DEV) for d in (d1, d2)]
            gR, gt = [torch.from_numpy(d1[k].astype(np.float64)).to(DEV) for k in ("global_R", "global_t")]
            gR2, gt2 = [torch.from_numpy(d2[k].astype(np.float64)).to(DEV) for k in ("global_R", "global_t")]
            for k in range(0, F, 500):
                sl = slice(k, min(k + 500, F)); B = sl.stop - sl.start
                m = torch.from_numpy(d1["mask"][sl].astype(bool)).to(DEV)
                R1 = sixd_rows_to_R(torch.from_numpy(d1["pose6d"][sl].astype(np.float64)).to(DEV))
                R2 = sixd_rows_to_R(torch.from_numpy(d2["pose6d"][sl].astype(np.float64)).to(DEV))
                obs = []
                for W, R, j, d, Ra, ta in ((W1, R1, j1, d1, gR, gt), (W2, R2, j2, d2, gR2, gt2)):
                    root = torch.from_numpy(d["root_position"][sl].astype(np.float64)).to(DEV)
                    L = lbs(P, W, R, j, parents, root) @ rot.T
                    obs.append(L @ Ra.T + ta + torch.from_numpy(d["residual_m"][sl].astype(np.float64)).to(DEV))
                c_off = torch.from_numpy(d2["root_offset"].astype(np.float64)).to(DEV)
                Mr = torch.from_numpy(d2["pose_rest_rotation"].astype(np.float64)).to(DEV)
                root_pred = torch.from_numpy(d1["root_position"][sl].astype(np.float64)).to(DEV) - (R1[:, 0] @ Mr @ c_off.unsqueeze(-1)).squeeze(-1)
                # [6] posed skeleton: raw pose on the shot's BVH skeleton vs V2 pose on the rig (rotated by M), root-aligned
                own = fk(R1, j1, parents)
                e_fk = max(e_fk, (own - fk(R2, j2, parents)).norm(dim=-1).max().item())   # G_rig = G_own M, own offsets = M o
                e_root = max(e_root, (root_pred - torch.from_numpy(d2["root_position"][sl].astype(np.float64)).to(DEV)).abs().max().item())
                # float16 storage: |error| <= half an ulp of both stored residuals (2^-11 relative each)
                tol = (torch.from_numpy(d1["residual_m"][sl].astype(np.float64)).to(DEV).norm(dim=-1)
                       + torch.from_numpy(d2["residual_m"][sl].astype(np.float64)).to(DEV).norm(dim=-1)) * 2.0 ** -10 + 2e-6
                e_obs = max(e_obs, ((obs[0] - obs[1]).norm(dim=-1) / tol)[m].max().item()); n_obs += int(m.sum())
                # packed pose: same local rotations as pose6d except the root (Z-up fix)
                Rp = sixd_cols_to_R(torch.from_numpy(p2["pose"][sl].astype(np.float64)).to(DEV).view(B, -1, 6))
                e_pk = max(e_pk, (Rp[:, 1:] - R2[:, 1:]).abs().max().item())
                # training target (model frame, stored units): V2 vs 01's V1 conversion
                Rp1 = sixd_cols_to_R(torch.from_numpy(p1["pose"][sl].astype(np.float64)).to(DEV).view(B, -1, 6))
                z = torch.zeros(B, 3, dtype=torch.float64, device=DEV)
                r1 = torch.from_numpy(d1["residual_scaled"][sl].astype(np.float64)).to(DEV) @ gR @ R2M.to(DEV).T
                Rp1_dt = detwist(Rp1, torch.from_numpy(rig["j_rest"].astype(np.float64)).to(DEV), names, parents) if S in DETWIST_SUBJECTS else Rp1
                t01 = r1 + 0.1 * (lbs(P, W1, Rp1, jc, parents, z) - lbs(P, W2, Rp1_dt, j2, parents, z))
                t2 = torch.from_numpy(p2["residuals"][sl].astype(np.float64)).to(DEV) @ gR2 @ R2M.to(DEV).T
                dt = (t2 - t01) * 10 * m.unsqueeze(-1)                                 # x10: stored units -> m
                dt = dt - dt.sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1).unsqueeze(-1)   # training centres per frame
                e_tgt.append(dt.norm(dim=-1)[m].cpu().numpy())
        e_tgt = np.concatenate(e_tgt) * 1000
        check(e_obs <= 1.0, f"observed markers identical V1 vs V2 over {n_obs} observations (max error / float16 bound {e_obs:.2f})")
        check(e_fk < 0.15, f"posed skeleton V2-on-rig vs raw-on-BVH, root-aligned: max {e_fk * 1000:.0f} mm (rig bone-length refit)")
        check(e_root < 1e-5, f"root_position == root_position_raw - R_root @ root_offset (max {e_root:.1e})")
        check(e_pk < 1e-5, f"packed pose == pose6d on non-root joints (max {e_pk:.1e})")
        print(f"  INFO V2 target vs 01's V1 conversion (centred): mean {e_tgt.mean():.2f} p99 {np.percentile(e_tgt, 99):.2f} max {e_tgt.max():.2f} mm", flush=True)
    print("ALL OK" if ok else "FAILURES (see above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
