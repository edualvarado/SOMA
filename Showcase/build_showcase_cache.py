"""Build the Blender showcase cache of one shot from `03_evaluate_metrics.py --export-showcase`.

Every vertex of the individual muscle meshes is bound to the SOMA muscle layer (T-pose) by a smooth Gaussian blend
of its k nearest layer vertices. Per frame:
  * muscles_tpose_d = blend of the muscle-layer displacement (canonical space; muscles = rest + d)
  * muscles_lbs     = LBS of the rest muscles with the blended skin weights and the model's per-joint transforms
                      (rigid correction included), i.e. the same posing as the muscle layer
  * muscles_posed_d = SOMA posed - muscles_lbs. LBS is linear in the vertex, so lbs + s * d is exactly the posed
                      muscles with the displacement scaled by s (s = 0: LBS baseline, s = 1: SOMA)
  * skin_tpose / skin_posed (float16, for a ghost skin), joints_posed (input motion)
The skeleton mesh moves rigidly: each connected component follows the joint that dominates its nearest bone-layer
vertices (static.npz bone_joint), so Blender only sets one matrix per joint and frame (A.npy).

Usage (from the repository root, SKIM_ROOT set; see Showcase/README.md):
    python Showcase/build_showcase_cache.py --model Showcase/cache/S1_shot_003_model.npz \
        --subject S1 --out Showcase/cache/S1_shot_003
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "05-Training"))
from muscle_binding import bind_to_layer  # noqa: E402


def load_obj(path):
    """Vertices (N,3) and polygons (list of index lists, quads kept) of an OBJ, in file order."""
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    return np.asarray(verts, dtype=np.float64), faces


def muscle_colors(names):
    """One hue per muscle, the same for its left/right pair (golden-ratio spacing)."""
    bases = sorted({n[2:] if n[:2] in ("l-", "r-") else n for n in names})
    hue = {b: (i * 0.61803398875) % 1.0 for i, b in enumerate(bases)}
    out = []
    for n in names:
        h = hue[n[2:] if n[:2] in ("l-", "r-") else n]
        i, f = int(h * 6), h * 6 - int(h * 6)
        s, v = 0.85, 1.0
        p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
        out.append([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6])
    return np.asarray(out, dtype=np.float32)


def lbs(v, W, rot, j_rest, parents, root):
    """Plain LBS of v (N,3) for every frame: rot (F,J,3,3), root (F,3) -> (F,N,3)."""
    n_frames, J = rot.shape[:2]
    G = np.zeros((n_frames, J, 4, 4))
    for i in range(J):
        L = np.zeros((n_frames, 4, 4)); L[:, :3, :3] = rot[:, i]; L[:, 3, 3] = 1
        L[:, :3, 3] = root if parents[i] < 0 else j_rest[i] - j_rest[parents[i]]
        G[:, i] = L if parents[i] < 0 else G[:, parents[i]] @ L
    G[..., :3, 3] -= np.einsum("fjab,jb->fja", G[..., :3, :3], j_rest)                     # @ inverse rest transform
    return np.einsum("nj,fjab,nb->fna", W, G, np.c_[v, np.ones(len(v))])[..., :3]


FLIP = np.array([-1.0, 1.0, -1.0])                     # dataset world (BVH Y-up, before the +90 deg X turn) -> capture


def rot6d(pose6d):
    """BVH 6D (row-major R[:, :2]) (F,J,6) -> (F,J,3,3)."""
    m = pose6d.reshape(*pose6d.shape[:2], 3, 2)
    x = m[..., 0] / np.linalg.norm(m[..., 0], axis=-1, keepdims=True)
    y = m[..., 1] - (x * m[..., 1]).sum(-1, keepdims=True) * x
    y /= np.linalg.norm(y, axis=-1, keepdims=True)
    return np.stack([x, y, np.cross(x, y)], -1)


def world_transforms(M, sh, s_bind, W, n=400):
    """Per-frame rigid transform (F,4,4) from the model frame (as cached) to the dataset's capture world (Y-up, m):
    flip(global_R @ LBS_bvh + global_t), flip = (-x, y, -z) (the dataset's Z-up world is (x, -z, y) of that sum,
    triangulated points map to it as (-x, z, y)). Both sides are plain LBS of the same skin vertices for the same
    pose, so the per-frame Kabsch fit is exact. The dataset world matches the calibration frame only up to a few cm
    (the per-shot global_R/t refit), soma_xray.py corrects it from markers_world.npz."""
    sel = np.random.default_rng(0).choice(len(s_bind), n, replace=False)
    v, w = s_bind[sel], W[sel]
    P = lbs(v, w, M["rot"], M["j_rest"], M["parents"], np.broadcast_to(M["j_rest"][0], (len(M["rot"]), 3)))
    Q = lbs(v, w, rot6d(sh["pose6d"]), sh["j_rest"], sh["parents"], sh["root_position"])
    Q = (Q @ sh["global_R"].T + sh["global_t"]) * FLIP
    pc, qc = P.mean(1, keepdims=True), Q.mean(1, keepdims=True)
    U, _, Vt = np.linalg.svd(np.swapaxes(P - pc, 1, 2) @ (Q - qc))
    D = np.ones((len(P), 3)); D[:, 2] = np.sign(np.linalg.det(np.swapaxes(Vt, 1, 2) @ np.swapaxes(U, 1, 2)))
    R = np.swapaxes(Vt, 1, 2) @ (D[:, :, None] * np.swapaxes(U, 1, 2))
    T = np.tile(np.eye(4), (len(P), 1, 1)); T[:, :3, :3] = R; T[:, :3, 3] = qc[:, 0] - np.einsum("fab,fb->fa", R, pc[:, 0])
    err = np.linalg.norm(np.einsum("fab,fnb->fna", R, P) + T[:, None, :3, 3] - Q, axis=-1).max()
    print(f"capture world: per-frame rigid fit, max residual {err * 1000:.3f} mm")
    return T.astype(np.float32)


def observed_markers(sh, can):
    """The dataset's observed markers (F,M,3) in its capture world (see world_transforms), NaN where unobserved."""
    x = lbs(can["p_bind"], can["W"], rot6d(sh["pose6d"]), sh["j_rest"], sh["parents"], sh["root_position"])
    x = (x @ sh["global_R"].T + sh["global_t"] + sh["residual_m"]) * FLIP
    x[sh["mask"] == 0] = np.nan
    return x.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="npz from 03_evaluate_metrics.py --export-showcase")
    ap.add_argument("--data-root", default=os.environ.get("SKIM_ROOT"), help="SKIM dataset root (default: $SKIM_ROOT)")
    ap.add_argument("--subject", default="S1")
    ap.add_argument("--out", required=True, help="output cache directory")
    ap.add_argument("--k", type=int, default=16, help="muscle-layer vertices blended per muscle vertex")
    ap.add_argument("--fps", type=float, default=25.0)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    M = dict(np.load(args.model))  # npz members are re-read on every access otherwise
    layers = os.path.join(args.data_root, args.subject, "layers", "tpose")
    m_bind, s_bind, b_bind, W = M["m_bind"], M["s_bind"], M["b_bind"], M["skin_weights"]
    # The model poses with the root at the origin: lift every transform by the rest hips so posed and T-pose stand alike
    M["A"][..., 3] += M["j_rest"][0]
    A = torch.from_numpy(M["A"]).float().to(dev)                                         # (F, J, 3, 4)
    n_frames, J = A.shape[:2]

    # Layer meshes: same vertex order as the model (03 loads them unprocessed)
    m_obj, _ = load_obj(os.path.join(layers, f"musc_layer-{args.subject}-TPose.obj"))
    s_obj, s_faces = load_obj(os.path.join(layers, f"skin_layer-{args.subject}-TPose.obj"))
    print(f"muscle layer vs model m_bind: max {np.abs(m_obj - m_bind).max():.2e} m (non-zero = LAYER_REST_OFFSET)")

    # ---- Individual muscles: binding to the muscle layer ----
    paths = sorted(glob.glob(os.path.join(layers, "muscle_meshes_tpose", "*.obj")))
    names, verts, faces, ranges = [], [], [], []
    off = m_bind[0] - m_obj[0]                       # LAYER_REST_OFFSET (0 except S4)
    for p in paths:
        v, f = load_obj(p)
        names.append(os.path.basename(p)[:-4].replace("_t-pose", ""))
        ranges.append((sum(len(x) for x in verts), sum(len(x) for x in verts) + len(v)))
        faces.extend([[i + ranges[-1][0] for i in poly] for poly in f])
        verts.append(v + off)
    rest = np.concatenate(verts)
    idx, w = bind_to_layer(rest, m_bind, k=args.k)
    print(f"{len(names)} muscles, {len(rest)} vertices bound to the muscle layer")
    idx_t, w_t = torch.from_numpy(idx).to(dev), torch.from_numpy(w).float().to(dev)
    rest_t = torch.from_numpy(rest).float().to(dev)
    Wm = (w_t.unsqueeze(-1) * torch.from_numpy(W).float().to(dev)[idx_t]).sum(1)           # (N, J) blended skin weights

    def pose(x, f0, f1, weights):
        """LBS of x (B,N,3) with the per-joint transforms of frames f0:f1."""
        hom = torch.cat([x, torch.ones_like(x[..., :1])], -1)
        T = torch.einsum("nj,bjac->bnac", weights, A[f0:f1])                                 # (B, N, 3, 4)
        return (T @ hom.unsqueeze(-1))[..., 0]

    N, V = len(rest), len(m_bind)
    out = {k: np.lib.format.open_memmap(os.path.join(args.out, f"{k}.npy"), mode="w+", dtype=dt, shape=sh) for k, dt, sh in [
        ("muscles_tpose_d", np.float16, (n_frames, N, 3)), ("muscles_lbs", np.float32, (n_frames, N, 3)),
        ("muscles_posed_d", np.float16, (n_frames, N, 3)),
        ("skin_tpose", np.float16, (n_frames, V, 3)), ("skin_posed", np.float16, (n_frames, V, 3))]}
    W_t, s_bind_t = torch.from_numpy(W).float().to(dev), torch.from_numpy(s_bind).float().to(dev)
    step = 32
    with torch.no_grad():
        for f0 in range(0, n_frames, step):
            f1 = min(f0 + step, n_frames)
            dm = torch.from_numpy(M["d_muscle"][f0:f1].astype(np.float32)).to(dev)            # (B, V, 3)
            dt = (w_t.unsqueeze(-1) * dm[:, idx_t]).sum(2)                                    # (B, N, 3)
            lbs = pose(rest_t.expand(f1 - f0, -1, -1), f0, f1, Wm)
            st = s_bind_t + torch.from_numpy(M["d_skin"][f0:f1].astype(np.float32)).to(dev)
            out["muscles_tpose_d"][f0:f1] = dt.cpu().numpy().astype(np.float16)
            out["muscles_lbs"][f0:f1] = lbs.cpu().numpy()
            out["muscles_posed_d"][f0:f1] = (pose(rest_t + dt, f0, f1, Wm) - lbs).cpu().numpy().astype(np.float16)
            out["skin_tpose"][f0:f1] = st.cpu().numpy().astype(np.float16)
            out["skin_posed"][f0:f1] = pose(st, f0, f1, W_t).cpu().numpy().astype(np.float16)
            print(f"\rframes {f1}/{n_frames}", end="")
    print()
    for a in out.values():
        a.flush()
    disp = np.linalg.norm(out["muscles_tpose_d"][::10].astype(np.float32), axis=-1)
    print(f"muscle displacement (T-pose): mean {disp.mean() * 1000:.2f} mm, p99 {np.percentile(disp, 99) * 1000:.2f} mm, "
          f"max {disp.max() * 1000:.2f} mm")

    # ---- Input motion: posed joints ----
    jr = torch.from_numpy(M["j_rest"]).float().to(dev)
    joints = (A[..., :3] @ jr.view(1, J, 3, 1))[..., 0] + A[..., 3]                          # (F, J, 3)
    np.save(os.path.join(args.out, "joints_posed.npy"), joints.cpu().numpy())
    np.save(os.path.join(args.out, "A.npy"), M["A"])
    shot_npz = os.path.join(args.data_root, args.subject, f"{M['shot']}.npz") if "shot" in M else ""
    if "rot" in M and os.path.isfile(shot_npz):                    # for overlays on the capture videos (soma_xray.py)
        sh = np.load(shot_npz)
        np.save(os.path.join(args.out, "world_T.npy"), world_transforms(M, sh, s_bind, W))
        np.savez(os.path.join(args.out, "markers_world.npz"), ids=sh["marker_ids"],
                 xyz=observed_markers(sh, np.load(os.path.join(args.data_root, args.subject, "canonical.npz"))))

    # ---- Skeleton mesh: rigid per joint ----
    k_v, k_f = load_obj(os.path.join(layers, f"skel_mesh-{args.subject}-TPose.obj"))
    k_v = k_v + off
    e = np.array([(p[i], p[(i + 1) % len(p)]) for p in k_f for i in range(len(p))])
    n_comp, comp = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(k_v),) * 2), directed=False)
    _, kidx = cKDTree(b_bind).query(k_v, k=8)
    vj = W[kidx].mean(1)                                                                       # (K, J)
    comp_joint = np.array([vj[comp == c].sum(0).argmax() for c in range(n_comp)])
    print(f"skeleton mesh: {len(k_v)} vertices, {n_comp} components on {len(set(comp_joint))} joints")

    np.savez(os.path.join(args.out, "static.npz"),
             muscle_rest=rest.astype(np.float32), muscle_faces=np.concatenate(faces), muscle_face_sizes=[len(p) for p in faces],
             muscle_ranges=np.asarray(ranges), muscle_names=np.asarray(names), muscle_colors=muscle_colors(names),
             skin_rest=s_bind.astype(np.float32), skin_faces=np.concatenate(s_faces), skin_face_sizes=[len(p) for p in s_faces],
             bone_rest=k_v.astype(np.float32), bone_faces=np.concatenate(k_f), bone_face_sizes=[len(p) for p in k_f],
             bone_joint=comp_joint[comp], j_rest=M["j_rest"], parents=M["parents"], joint_names=M["joint_names"],
             skin_weights=W.astype(np.float32), muscle_weights=Wm.cpu().numpy())
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"subject": args.subject, "frames": int(n_frames), "fps": args.fps, "muscles": len(names),
                   "muscle_vertices": int(N), "model": os.path.basename(args.model), "up_axis": "Y"}, f, indent=1)
    print(f"cache -> {args.out}")


if __name__ == "__main__":
    main()
