"""Rebuild a subject's rig (rest joints + skin weights) from its own anatomy.

  1. Joint refit: each limb joint is moved to the contact between its two bones in the skeleton mesh
     (layers/tpose/skel_mesh-<S>-TPose.obj; e.g. elbow = humerus <-> radius/ulna). Spine, neck, head and
     clavicle joints keep the Captury rest positions.
  2. Skin weights by bone heat (Baran & Popovic 2007, the method behind Blender's "Automatic Weights"):
     (K + M H) w_j = M H p_j on the skin layer, K = cotangent stiffness, M = lumped mass, H_vv = 1/d(v)^2
     to the nearest bone segment, p_j = 1 where bone j is the nearest. Every vertex (hands included) gets
     weights that sum to 1. Skin, muscle and bone layers share the topology, so they share the weights.

Output: <out>/<S>_rig.npz with j_rest (J,3) in the BVH rest frame, weights (V,J), joint_names, and the
original j_rest for reference. Loaded by 01/02/03 with --rig.

Usage (from 05-Training/):  python build_rig.py --subject S1 --data-root $SKIM_ROOT [--out rigs]
"""
import argparse
import glob
import os

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh
from pymotion.io.bvh import BVH
from scipy.spatial import cKDTree

# Joints refitted from the bone contacts (the rest keep the Captury position). The shoulder (Arm) is left out:
# scapula/clavicle/ribs make its contact ambiguous, and Captury's is within ~1.5 cm of the limb axis.
REFIT = ["LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg", "LeftFoot", "RightFoot",
         "LeftForeArm", "RightForeArm", "LeftHand", "RightHand"]
# Limb bones only heat vertices beyond their proximal joint (minus this margin, m)
LIMB_KEYS = ("UpLeg", "Leg", "Foot", "ToeBase", "Arm", "ForeArm", "Hand")
HALF_SPACE_MARGIN = 0.02


def load_obj(path):
    V, F = [], []
    for line in open(path):
        if line.startswith("v "):
            V.append(line.split()[1:4])
        elif line.startswith("f "):
            F.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.asarray(V, np.float64), np.asarray(F, np.int64)


def rest_skeleton(bvh_path, hips_offset=None, scale=1.0):
    bvh = BVH(); bvh.load(bvh_path)
    _, _, parents, offsets, _, _ = bvh.get_data()
    parents = parents.copy(); parents[0] = -1
    offsets = offsets * scale
    j = np.zeros_like(offsets)
    for i in range(len(parents)):
        j[i] = offsets[i] if parents[i] < 0 else j[parents[i]] + offsets[i]
    if hips_offset is not None:
        j = j + np.asarray(hips_offset)
    return j, parents, [str(n) for n in bvh.data["names"]]


def end_tip(j, parents, i, V):
    """Tip of an end joint's bone: the farthest skin vertex along the parent->joint direction, within a cone."""
    d = j[i] - j[parents[i]]; d /= np.linalg.norm(d)
    rel = V - j[i]; along = rel @ d
    radial = np.linalg.norm(rel - along[:, None] * d, axis=1)
    cone = (along > 0) & (radial < 0.6 * along + 0.03)
    return j[i] + d * along[cone].max()


def bone_segments(j, parents, V):
    """Weight column i <- segments from joint i to each child (end joints: to their tip)."""
    segs = []
    for i in range(len(parents)):
        children = [k for k in range(len(parents)) if parents[k] == i]
        for k in children:
            segs.append((i, j[i], j[k]))
        if not children:
            segs.append((i, j[i], end_tip(j, parents, i, V)))
    return segs


def point_segment_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def refit_joints(j, parents, names, skel_V, skel_F, skin_V):
    """Move each REFIT joint onto the axis through the contact between its parent and child bones.

    Bones (skeleton-mesh components) are assigned to the closest segment whose span contains their centroid
    (keeps the hip bones off the femur and the patella off the tibia). The joint centre is the mean of the
    contact-patch centres of the parent bones that touch the child bones (ankle: tibia and fibula on the
    talus). Only the component perpendicular to the parent bone is applied: the contact lies on the joint
    surface, not on the rotation axis, and Captury's bone lengths were fitted to the markers."""
    comps = trimesh.Trimesh(skel_V, skel_F, process=False).split(only_watertight=False)
    segs = bone_segments(j, parents, skin_V)
    owner = []
    for c in comps:
        cen, best, o = c.vertices.mean(0), np.inf, -1
        for jid, a, b in segs:
            t = ((cen - a) @ (b - a)) / max((b - a) @ (b - a), 1e-12)
            d = point_segment_dist(c.vertices, a, b).mean()
            if -0.05 <= t <= 1.05 and d < best:
                best, o = d, jid
        owner.append(o)
    owner = np.asarray(owner)
    j_new = j.copy()
    for n in REFIT:
        i = names.index(n)
        par = [c.vertices for c, o in zip(comps, owner) if o == parents[i]]
        chi = [c.vertices for c, o in zip(comps, owner) if o == i]
        if not par or not chi:
            print(f"  {n}: no bones on one side, kept")
            continue
        tree = cKDTree(np.concatenate(chi))
        gaps, centres = [], []
        for P in par:
            d, idx = tree.query(P)
            close = np.argsort(d)[:100]  # this bone's contact patch
            gaps.append(d[close].mean())
            centres.append(0.5 * (P[close] + tree.data[idx[close]]).mean(0))
        gaps = np.asarray(gaps)
        touch = gaps <= max(1.5 * gaps.min(), gaps.min() + 0.01)
        target = np.mean(np.asarray(centres)[touch], axis=0)
        axis = (j[i] - j[parents[i]]) / np.linalg.norm(j[i] - j[parents[i]])
        delta = target - j[i]
        delta = delta - (delta @ axis) * axis
        j_new[i] = j[i] + delta
        print(f"  {n:14s} moved {np.round(delta * 100, 1)} cm ⟂ bone  (contact gap {gaps[touch].mean() * 100:.1f} cm, "
              f"{int(touch.sum())}/{len(par)} parent bones touching, {len(chi)} child bones)")
    return j_new


def cotan_laplacian(V, F):
    """Cotangent stiffness K (PSD) and lumped (barycentric) mass M."""
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]
    e0, e1, e2 = V[i2] - V[i1], V[i0] - V[i2], V[i1] - V[i0]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    def cot(a, b):
        return (a * b).sum(1) / (np.linalg.norm(np.cross(a, b), axis=1) + 1e-12)
    c0, c1, c2 = cot(-e1, e2), cot(-e2, e0), cot(-e0, e1)  # angle at vertex 0, 1, 2
    rows = np.concatenate([i1, i2, i2, i0, i0, i1])
    cols = np.concatenate([i2, i1, i0, i2, i1, i0])
    vals = 0.5 * np.concatenate([c0, c0, c1, c1, c2, c2])
    W = sp.coo_matrix((vals, (rows, cols)), shape=(len(V), len(V))).tocsr()
    K = sp.diags(np.asarray(W.sum(1)).ravel()) - W
    mass = np.zeros(len(V)); np.add.at(mass, F.ravel(), np.repeat(area / 3.0, 3))
    return K, sp.diags(mass)


def bone_heat(V, F, j, parents, names, prune=0.01, max_inf=0, pitch=0.004):
    """Bone heat with Pinocchio's visibility test (a vertex is heated only by a bone it sees through the inside of
    the body: keeps the torso flanks off the arms) and a joint half-space rule for limb bones (a limb bone heats
    only vertices beyond its proximal joint: keeps the belly off the thighs)."""
    segs = bone_segments(j, parents, V)
    J = len(parents)
    A = np.stack([a for _, a, _ in segs]); Bv = np.stack([b for _, _, b in segs])
    seg_joint = np.array([s[0] for s in segs])
    AB = Bv - A
    t = np.einsum('vsk,sk->vs', V[:, None] - A[None], AB) / np.maximum((AB ** 2).sum(1), 1e-12)
    Q = A[None] + np.clip(t, 0, 1)[..., None] * AB[None]                    # nearest point on each bone (V,S,3)
    D = np.linalg.norm(V[:, None] - Q, axis=2)
    limb = np.array([any(k in names[i] for k in LIMB_KEYS) for i in seg_joint])
    seg_len = np.linalg.norm(AB, axis=1)
    D[:, limb] = np.where(t[:, limb] * seg_len[limb] >= -HALF_SPACE_MARGIN, D[:, limb], np.inf)
    # visibility: samples along vertex -> bone point must be inside the body (voxelised skin, filled)
    vox = trimesh.Trimesh(V, F, process=False).voxelized(pitch).fill()
    occ, inv = vox.matrix, np.linalg.inv(vox.transform)
    def inside(P):
        idx = np.round(P @ inv[:3, :3].T + inv[:3, 3]).astype(int)
        ok = ((idx >= 0) & (idx < occ.shape)).all(-1)
        out = np.zeros(P.shape[:-1], bool)
        out[ok] = occ[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
        return out
    s = np.linspace(0.2, 1.0, 12)
    order = np.argsort(D, axis=1)[:, :6]
    vis = np.zeros_like(D, bool)
    for c in range(order.shape[1]):
        k = order[:, c]
        q = Q[np.arange(len(V)), k]
        samples = V[:, None] + s[None, :, None] * (q - V)[:, None]
        vis[np.arange(len(V)), k] = inside(samples).all(1) & np.isfinite(D[np.arange(len(V)), k])
    Dv = np.where(vis, D, np.inf)
    dmin = Dv.min(1)
    seen = np.isfinite(dmin)
    print(f"  visibility: {seen.mean() * 100:.1f}% of vertices see a bone; "
          f"{(seen & (Dv.argmin(1) != D.argmin(1))).sum()} switched away from the nearest bone")
    # nearest visible bone(s) heat the vertex; vertices that see none get no heat (pure diffusion, as Pinocchio)
    near = seen[:, None] & (Dv <= dmin[:, None] * (1 + 1e-4))
    H = np.where(seen, 1.0 / np.maximum(dmin, 1e-3) ** 2, 0.0)
    P = np.zeros((len(V), J))
    for si in range(len(segs)):
        P[near[:, si], seg_joint[si]] = 1.0
    P[seen] /= P[seen].sum(1, keepdims=True)
    K, M = cotan_laplacian(V, F)
    MH = M @ sp.diags(H)
    lu = spla.splu((K + MH).tocsc())
    Wt = lu.solve(MH @ P)
    Wt = np.clip(Wt, 0.0, 1.0)
    Wt[Wt < prune] = 0.0
    if max_inf:
        keep = np.argsort(-Wt, axis=1)[:, :max_inf]
        mask = np.zeros_like(Wt, bool); np.put_along_axis(mask, keep, True, axis=1)
        Wt[~mask] = 0.0
    return Wt / Wt.sum(1, keepdims=True)


def blend_with_original(W_new, W_old, V, F, names, smooth_iters=40):
    """Original (artist) weights on the torso, bone-heat weights on the limbs: pure bone heat lets the arm/thigh
    weights diffuse far into the flanks and belly (all torso bones are far from the skin), which thins the torso
    when the arms go down or the hips flex. alpha = 1 where the original dominant joint is a limb bone or the
    original row is empty (hands), 0 on the torso, smoothed over the mesh for the transition."""
    limb = [i for i, n in enumerate(names) if any(k in n for k in LIMB_KEYS)]
    alpha = (np.isin(W_old.argmax(1), limb) | (W_old.sum(1) < 0.99)).astype(np.float64)
    E = np.unique(np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1), axis=0)
    A = sp.coo_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(len(V), len(V))).tocsr()
    A = sp.diags(1.0 / np.asarray(A.sum(1)).ravel()) @ A
    for _ in range(smooth_iters):
        alpha = A @ alpha
    W = alpha[:, None] * W_new + (1 - alpha[:, None]) * W_old
    return W / W.sum(1, keepdims=True), alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S1")
    ap.add_argument("--data-root", default=os.environ.get("SKIM_ROOT"))
    ap.add_argument("--out", default="rigs")
    ap.add_argument("--no-refit", action="store_true", help="Keep the Captury rest joints (weights only)")
    ap.add_argument("--pure-heat", action="store_true", help="Bone heat everywhere (default: original weights on the torso)")
    a = ap.parse_args()
    S, base = a.subject, os.path.join(a.data_root, a.subject)
    hips = {"S2": (-0.002, 0.876, -0.075), "S3": (0.0, 0.889, -0.129), "S4": (0.0, 0.849, -0.083),
            "S5": (0.007, 0.839, -0.065)}.get(S)  # same HIPS_REST_OFFSET as 01/02/03
    bvh_path = sorted(glob.glob(os.path.join(base, "raw", "shot_*_captury", "*.bvh")))[0]
    j, parents, names = rest_skeleton(bvh_path, hips, scale=1.0 if S == "S1" else 0.001)  # S2-S5 BVHs are in mm (as 01)
    skin_V, skin_F = load_obj(os.path.join(base, "layers", "tpose", f"skin_layer-{S}-TPose.obj"))
    print(f"[{S}] skeleton from {os.path.basename(bvh_path)}, skin {len(skin_V)} verts")
    bone_V, bone_F = load_obj(os.path.join(base, "layers", "tpose", f"skel_mesh-{S}-TPose.obj"))
    bone_V = bone_V + np.asarray({"S4": (0.002098, 0.014745, 0.010864)}.get(S, (0.0, 0.0, 0.0)))  # LAYER_REST_OFFSET of 01/02/03
    j_new = j if a.no_refit else refit_joints(j, parents, names, bone_V, bone_F, skin_V)
    W = bone_heat(skin_V, skin_F, j_new, parents, names)
    if not a.pure_heat:
        W_old = np.load(os.path.join(base, "canonical_model", "lbs_skin", f"{S}_skin_lbs_weights_exported.npy"))
        W, alpha = blend_with_original(W, W_old, skin_V, skin_F, names)
        print(f"[{S}] torso from the original weights: {(alpha < 0.05).mean() * 100:.0f}% original, "
              f"{(alpha > 0.95).mean() * 100:.0f}% bone heat, {((alpha >= 0.05) & (alpha <= 0.95)).mean() * 100:.0f}% blended")
    print(f"[{S}] weights: rows sum {W.sum(1).min():.3f}..{W.sum(1).max():.3f}, "
          f"mean influences {(W > 0).sum(1).mean():.2f}")
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"{S}_rig.npz")
    np.savez(out, j_rest=j_new.astype(np.float32), weights=W.astype(np.float32), joint_names=np.array(names),
             j_rest_captury=j.astype(np.float32))
    print(f"[{S}] saved {out}")


if __name__ == "__main__":
    main()
