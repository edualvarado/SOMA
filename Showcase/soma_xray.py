"""X-ray overlay: SOMA's posed muscles and bones on the capture video, revealed through a lens that follows the body.

    blender --python Showcase/soma_xray.py -- --cache Showcase/cache/S1_shot_003 \
        --video <capture>/shot_003/stream001.mp4 --calib <capture>/shot_003/cameras.calib --camera 1 \
        [--markers <S>_triangulated_sequence_<shot>_transformed_filtered.json \
         --detections <capture>/shot_003/detections_charuco-suit/corners_markers_stream001.json] \
        [--offset 0 0] [--depth 0] [--frames 500 750] [--lens-joint Spine2] [--lens-radius 0.2] [--lens-open 1] [--lens-delay 0] \
        [--color heatmap] [--disp-max 10] [--scale 0.5] [--style retro] [--crop X Y W H] \
        [--face-landmarks <landmarks>.npz] [--blur-face --face-boxes <faces>.npz] [--skull-mesh <skull>.obj] \
        [--save Showcase/SOMA_xray.blend] [--render Showcase/xray.mp4]

Needs a cache with world_T.npy (build_showcase_cache.py) and the raw capture video + camera calibration of the shot
(not part of SKIM). The dataset's world matches the calibration frame only up to a few cm, which shows on a close-up
camera: --markers (the raw triangulation) fits one rigid correction per shot from the dataset's observed markers
(markers_world.npz), on the markers --detections finds in this camera. The calibrated camera (intrinsics incl. principal point, extrinsics) is reproduced in Blender and
the characters are placed in the capture world per frame (keyframes), so the overlay needs no runtime code beyond the
showcase player stored in the .blend. Compositor: video outside the lens; inside, the darkened video with the muscles
and bones on top, a soft edge and a thin ring. The lens opens over --lens-open seconds (after --lens-delay seconds of
plain video); with --color heatmap the muscles show |SOMA - LBS| and a colour bar (a second scene) sits in the upper-left corner.

The head has no markers: --face-landmarks (5 face points per frame from a face detector) turns the neck and head a
little towards the face in the video. --blur-face blurs the face in the video only, so the skull drawn over it stays
sharp. --style retro: palette muscles, x-ray shaded bones, blue-tinted lens with scan lines and a HUD ring that draws
itself on while the lens opens. --skull-mesh swaps the skull for another mesh in the subject's rest frame (credit its
source if you publish the result).
"""
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Quaternion

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import soma_showcase as ss  # noqa: E402  (registers the player: panel + frame handler)

Y_UP_TO_Z_UP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)   # capture world -> Blender
CV_TO_BLENDER_CAM = np.diag([1.0, -1.0, -1.0, 1.0])                                        # OpenCV -> Blender camera axes
# skin vertices (shared SKIM topology) whose mean is each YuNet landmark: right eye, left eye, nose tip, right / left
# mouth corner (subject's sides)
FACE_VERTS = ([463, 459, 460, 521], [23572, 11139, 23566, 11077], [22471], [697], [11328])


def read_camera(calib, idx, width):
    """World->camera extrinsics E (4,4, m) and pixel intrinsics K (3,3) of camera idx in a cameras.calib file."""
    lines = open(calib).read().splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(f"camera\t{idx}\t"))
    blk = lines[i:i + 25]
    E = np.eye(4)
    E[:3] = [[float(v) for v in blk[r].split()[1:]] for r in (16, 17, 18)]
    E[:3, 3] /= 1000.0                                                                      # mm -> m
    K = np.array([[float(v) for v in blk[r].split()[1:]] for r in (20, 21, 22)])
    K[:2] *= width                                                                          # stored relative to the width
    return E, K


def kabsch(P, Q, w=None):
    """Rigid (4,4) minimising the (weighted) squared distance of P @ R.T + t to Q."""
    w = np.ones(len(P)) if w is None else w
    pc, qc = w @ P / w.sum(), w @ Q / w.sum()
    U, _, Vt = np.linalg.svd(((P - pc) * w[:, None]).T @ (Q - qc))
    R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    C = np.eye(4); C[:3, :3], C[:3, 3] = R, qc - R @ pc
    return C


def triangulation(markers, ids, f0, f1):
    """Raw triangulated markers (f1-f0+1, M, 3) of cache frames f0..f1 in the capture world, NaN where missing."""
    col = {str(n): j for j, n in enumerate(ids)}
    tri = json.load(open(markers))
    Q = np.full((f1 - f0 + 1, len(ids), 3), np.nan)
    for f in range(f0, f1 + 1):
        for name, x in tri.get(str(f + 1), {}).items():                                    # keyed by frame + 1
            j = col.get(name)
            if j is not None and x:
                Q[f - f0, j] = (-x[0][0], x[0][2], x[0][1])                                 # Z-up -> capture world
    return Q


def capture_correction(cache, Q, detections, f0, f1):
    """Rigid transform (4,4) from the dataset's capture world to the triangulation Q (calibration frame), fitted on the
    observed markers of cache frames f0..f1 (only those detected in the camera if `detections` is given)."""
    M = np.load(os.path.join(cache, "markers_world.npz"))
    P = M["xyz"][f0:f1 + 1]
    ok = ~np.isnan(P[..., 0]) & ~np.isnan(Q[..., 0])
    if detections:
        col, det, seen = {str(n): j for j, n in enumerate(M["ids"])}, json.load(open(detections)), np.zeros_like(ok)
        for f in range(f0, f1 + 1):
            for i in (det.get(str(f)) or {}).get("id_markers") or []:                       # keyed by video frame
                for c in range(4):
                    j = col.get(f"marker_{i[0]}_0_{c}")
                    if j is not None:
                        seen[f - f0, j] = True
        ok &= seen
    P, Q = P[ok], Q[ok]
    C = kabsch(P, Q)
    d0, d1 = np.linalg.norm(Q - P, axis=1), np.linalg.norm(Q - P @ C[:3, :3].T - C[:3, 3], axis=1)
    print(f"[XRAY] capture correction from {len(P)} markers: {np.degrees(np.arccos(min(1, (np.trace(C[:3, :3]) - 1) / 2))):.2f} deg, "
          f"mean offset {np.round((Q - P).mean(0) * 1000, 1)} mm, median |d| {np.median(d0) * 1000:.1f} -> "
          f"{np.median(d1) * 1000:.1f} mm")
    return C


def bone_correction(cache, T0, C, Q, f0, f1, smooth=9, min_markers=20, max_angle=30.0):
    """Per-joint rigid corrections D (f1-f0+1, J, 4, 4) in the calibration frame that move SOMA's skin at the markers onto
    the triangulated markers Q: the body follows the capture's pose (Captury BVH), which is a few cm off per limb.
    Weighted Kabsch per joint (weights = skinning weights of each marker's skin vertex) where at least `min_markers`
    markers belong mostly to the joint; the forearm twist is dropped (v2 removes it from the pose on purpose); joints
    without a fit, or whose fit turns more than `max_angle` deg from their parent's, take the parent's; smoothed over
    `smooth` frames. T0: model -> dataset capture world, C: capture correction."""
    S = np.load(os.path.join(cache, "static.npz"))
    W, par, names = S["skin_weights"], S["parents"], [str(n) for n in S["joint_names"]]
    n, J = f1 - f0 + 1, len(names)
    skin = np.load(os.path.join(cache, "skin_posed.npy"), mmap_mode="r")[f0:f1 + 1].astype(np.float32)
    skin = np.einsum("fab,fnb->fna", T0[f0:f1 + 1, :3, :3], skin) + T0[f0:f1 + 1, None, :3, 3]   # dataset world
    obs = np.load(os.path.join(cache, "markers_world.npz"))["xyz"][f0:f1 + 1]
    seen = ~np.isnan(obs[..., 0])
    nn = np.full(obs.shape[1], -1)
    for m in np.where(seen.any(0))[0]:                            # marker -> nearest skin vertex, mid observed frame
        i = np.where(seen[:, m])[0][seen[:, m].sum() // 2]
        nn[m] = np.argmin(((skin[i] - obs[i, m]) ** 2).sum(1))
    sk = skin[:, np.maximum(nn, 0)] @ C[:3, :3].T + C[:3, 3]                              # calibration frame
    wm = W[np.maximum(nn, 0)] * (nn >= 0)[:, None]
    jw = np.einsum("fab,fjb->fja", (C @ T0[f0:f1 + 1])[:, :3, :3], np.load(os.path.join(cache, "joints_posed.npy"))[f0:f1 + 1]) \
        + (C @ T0[f0:f1 + 1])[:, None, :3, 3]
    dom = np.where(nn >= 0, wm.argmax(1), -1)
    D, wt = np.tile(np.eye(4), (n, J, 1, 1)), np.zeros((n, J))
    for i in range(n):
        ok = ~np.isnan(Q[i, :, 0]) & (nn >= 0)
        for j in range(J):
            if (dom[ok] == j).sum() >= min_markers:
                D[i, j], wt[i, j] = kabsch(sk[i, ok], Q[i, ok], wm[ok, j]), wm[ok, j].sum()
        for side in ("Left", "Right"):                                                      # swing only
            j, c = names.index(f"{side}ForeArm"), names.index(f"{side}Hand")
            q = Matrix(D[i, j, :3, :3].tolist()).to_quaternion()
            a = (jw[i, c] - jw[i, j]) / np.linalg.norm(jw[i, c] - jw[i, j])
            tw = Quaternion((q.w, *(np.dot(q[1:], a) * a))).normalized()
            Rs, p = np.array((q @ tw.inverted()).to_matrix()), jw[i, j]                  # keep the elbow in place
            D[i, j, :3, 3] += D[i, j, :3, :3] @ p - Rs @ p
            D[i, j, :3, :3] = Rs
        for j in range(J):                                        # parents come first
            if par[j] >= 0 and (wt[i, j] == 0 or np.trace(D[i, j, :3, :3] @ D[i, par[j], :3, :3].T)
                                < 1 + 2 * np.cos(np.radians(max_angle))):
                D[i, j], wt[i, j] = D[i, par[j]], 1e-3
    out = D.copy()
    for i in range(n):
        s = slice(max(0, i - smooth // 2), i + smooth // 2 + 1)
        for j in range(J):
            w = wt[s, j] + 1e-9
            U, _, Vt = np.linalg.svd(np.einsum("f,fab->ab", w, D[s, j, :3, :3]))
            out[i, j, :3, :3], out[i, j, :3, 3] = U @ Vt, w @ D[s, j, :3, 3] / w.sum()
    ok = ~np.isnan(Q[..., 0]) & (nn >= 0)[None]
    X = np.einsum("mj,fjab,fmb->fma", wm, out[..., :3, :3], sk) + np.einsum("mj,fja->fma", wm, out[..., :3, 3])
    print(f"[XRAY] bone correction: median |d| {np.median(np.linalg.norm(sk - Q, axis=-1)[ok]) * 1000:.1f} -> "
          f"{np.median(np.linalg.norm(X - Q, axis=-1)[ok]) * 1000:.1f} mm")
    return out


def bone_rig(root, cache, Dm, f0):
    """Armature under the posed character applying the per-joint corrections Dm (F, J, 4, 4, model frame) on top of the
    player's per-frame meshes: muscles with their skinning weights, each skeleton bone rigidly with its joint."""
    S = np.load(os.path.join(cache, "static.npz"))
    J = Dm.shape[1]
    arm = bpy.data.objects.new("SOMA_Correction", bpy.data.armatures.new("SOMA_Correction"))
    root.users_collection[0].objects.link(arm)
    arm.parent = root
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for j in range(J):                                                          # rest = identity: pose = correction
        eb = arm.data.edit_bones.new(f"J{j:02d}")
        eb.head, eb.tail, eb.roll = (0, 0, 0), (0, 0.1, 0), 0.0
    bpy.ops.object.mode_set(mode="OBJECT")
    for i in range(len(Dm)):
        for j, pb in enumerate(arm.pose.bones):
            pb.matrix_basis = Matrix(Dm[i, j].tolist())
            pb.keyframe_insert("location", frame=f0 + i)
            pb.keyframe_insert("rotation_quaternion", frame=f0 + i)
    mus = bpy.data.objects["SOMA_Posed_Muscles"]
    for j, w in enumerate(S["muscle_weights"].T):
        vg = mus.vertex_groups.new(name=f"J{j:02d}")
        for v in np.where(w > 1e-4)[0]:
            vg.add([int(v)], float(w[v]), "REPLACE")
    targets = [mus]
    for j in range(J):
        b = bpy.data.objects.get(f"SOMA_Posed_Bone_{j:02d}")
        if b is not None:
            b.vertex_groups.new(name=f"J{j:02d}").add(range(len(b.data.vertices)), 1.0, "REPLACE")
            targets.append(b)
    for ob in targets:
        ob.modifiers.new("Correction", "ARMATURE").object = arm


def face_points(cache, f0, f1):
    """SOMA's posed face points (f1-f0+1, 5, 3; model frame) of scene frames f0..f1, in FACE_VERTS order."""
    skin = np.load(os.path.join(cache, "skin_posed.npy"), mmap_mode="r")[f0 - 1:f1]
    return np.stack([skin[:, v].astype(np.float64).mean(1) for v in FACE_VERTS], 1)


def project(X, E, K):
    xc = X @ E[:3, :3].T + E[:3, 3]
    uv = xc @ K.T
    return uv[..., :2] / uv[..., 2:]


def rigid(x, p):
    """(4,4) rotation by the axis-angle x[:3] about the point p, then translation x[3:]."""
    G = np.eye(4)
    G[:3, :3] = np.array(Matrix.Rotation(np.linalg.norm(x[:3]), 3, x[:3] if np.linalg.norm(x[:3]) > 0 else (1, 0, 0)))
    G[:3, 3] = p - G[:3, :3] @ p + x[3:]
    return G


def face_correction(cache, Hh, landmarks, E, K, f0, f1, shift=(0, 0), sigma_r=0.17, smooth=9, min_score=0.7,
                    max_jump=15.0):
    """Per scene frame f0..f1: rotation G (f1-f0+1, 4, 4, calibration world) about the Neck joint that turns the neck and
    head so SOMA's face points move towards the face landmarks detected in the video (npz: frames, lm (F, 5, 2) video
    px, NaN = none; `shift` = crop corner). The head has no markers, so its pose is the capture's (Captury); the
    rotation is kept small (prior sigma_r rad); weak detections (score < min_score) and fits more than max_jump deg
    from the median of their neighbours are dropped, the gaps interpolated and the result smoothed over `smooth`
    frames.
    Hh: (F, 4, 4) current head placement, model -> calibration world."""
    n = f1 - f0 + 1
    X = face_points(cache, f0, f1)
    names = [str(x) for x in np.load(os.path.join(cache, "static.npz"))["joint_names"]]
    piv = np.load(os.path.join(cache, "joints_posed.npy"))[f0 - 1:f1, names.index("Neck")]
    Hr = Hh[f0 - 1:f1]
    X = np.einsum("fab,fkb->fka", Hr[:, :3, :3], X) + Hr[:, None, :3, 3]
    piv = np.einsum("fab,fb->fa", Hr[:, :3, :3], piv) + Hr[:, :3, 3]
    B = np.load(landmarks)
    U = np.full((n, 5, 2), np.nan)
    rows = np.arange(n) + f0 - 1                                                            # cache frames
    m = (rows >= B["frames"][0]) & (rows <= B["frames"][-1])
    U[m] = B["lm"][rows[m] - B["frames"][0]] - np.asarray(shift, float)
    if "score" in B:
        sc = np.zeros(n); sc[m] = B["score"][rows[m] - B["frames"][0]]
        U[sc < min_score] = np.nan
    ok = ~np.isnan(U[:, 0, 0])
    if not ok.any():
        print("[XRAY] face fit: no landmarks")
        return np.tile(np.eye(4), (n, 1, 1))
    e0 = np.linalg.norm(project(X, E, K) - U, axis=-1)
    if np.nanmedian(np.linalg.norm(project(X, E, K) - U[:, [1, 0, 2, 4, 3]], axis=-1)) < np.nanmedian(e0):
        U = U[:, [1, 0, 2, 4, 3]]                                                          # detector's sides swapped
    x = np.zeros((n, 3))
    for i in np.where(ok)[0]:
        sig = 0.05 * (np.linalg.norm(U[i, 0] - U[i, 4]) + np.linalg.norm(U[i, 1] - U[i, 3])) / 2   # px

        def res(y):
            G = rigid(np.r_[y, 0, 0, 0], piv[i])
            return np.r_[((project(X[i] @ G[:3, :3].T + G[:3, 3], E, K) - U[i]) / sig).ravel(), y / sigma_r]
        y = np.zeros(3)
        for _ in range(10):                                                                 # Gauss-Newton
            r = res(y)
            Jm = np.stack([(res(y + d) - r) / 1e-6 for d in np.eye(3) * 1e-6], 1)
            y = y - np.linalg.lstsq(Jm, r, rcond=None)[0]
        x[i] = y
    t = np.arange(n)
    good = ok.copy()
    for i in np.where(ok)[0]:                                                               # spikes vs the neighbours
        nb = ok & (np.abs(t - i) <= smooth) & (t != i)
        if nb.sum() >= 3 and np.degrees(np.linalg.norm(x[i] - np.median(x[nb], 0))) > max_jump:
            good[i] = False
    ok = good
    x = np.stack([np.interp(t, t[ok], x[ok, d]) for d in range(3)], 1)
    k = np.ones(smooth) / smooth
    x = np.stack([np.convolve(np.pad(x[:, d], smooth // 2, mode="edge"), k, mode="valid") for d in range(3)], 1)
    G = np.stack([rigid(np.r_[x[i], 0, 0, 0], piv[i]) for i in range(n)])
    e2 = np.linalg.norm(project(np.einsum("fab,fkb->fka", G[:, :3, :3], X) + G[:, None, :3, 3], E, K) - U, axis=-1)
    print(f"[XRAY] face fit ({ok.sum()}/{n} frames with landmarks): landmark error median {np.nanmedian(e0):.0f} -> "
          f"{np.nanmedian(e2):.0f} px, neck rotation median {np.degrees(np.median(np.linalg.norm(x, axis=1))):.1f} deg")
    return G


def replace_skull(cache, path):
    """Swap the posed skull's geometry for an OBJ given in the subject's rest frame (the bone objects keep their rest
    coordinates; the player poses them by object transform). Its UVs are kept, and maps next to it (<stem>_normal.png,
    <stem>_ao.png) are remembered on the mesh for retro_bones."""
    V, VT, F, FT = [], [], [], []
    for line in open(path):
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vt "):
            VT.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("f "):
            ids = [t.split("/") for t in line.split()[1:]]
            F.append([int(i[0]) - 1 for i in ids])
            FT.append([int(i[1]) - 1 for i in ids] if len(ids[0]) > 1 and ids[0][1] else [])
    names = [str(n) for n in np.load(os.path.join(cache, "static.npz"))["joint_names"]]
    ob = bpy.data.objects[f"SOMA_Posed_Bone_{names.index('Head'):02d}"]
    # the head bone also holds the upper cervical vertebrae and discs: keep those small parts of the original mesh
    OV = np.array([v.co[:] for v in ob.data.vertices])
    OF = [list(p.vertices) for p in ob.data.polygons]
    e = np.concatenate([np.c_[f, np.roll(f, 1)] for f in map(np.array, OF)])
    lab = np.arange(len(OV))                                    # connected parts: min-label propagation
    while True:
        m = np.minimum(lab[e[:, 0]], lab[e[:, 1]])
        new = lab.copy(); np.minimum.at(new, e[:, 0], m); np.minimum.at(new, e[:, 1], m); new = new[new]
        if (new == lab).all():
            break
        lab = new
    small = np.bincount(lab, minlength=len(OV))[lab] < 0.1 * len(OV)     # cranium, mandible: the large parts
    idx = -np.ones(len(OV), int); idx[small] = len(V) + np.arange(small.sum())
    neck = [[int(idx[i]) for i in f] for f in OF if small[f[0]]]
    V += OV[small].tolist(); F += neck
    if VT and all(FT):
        FT += [[len(VT)] * len(f) for f in neck]; VT.append([0.0, 0.0])
    me = bpy.data.meshes.new("SOMA_Skull")
    me.from_pydata(V, [], F)
    if VT and all(FT):
        me.uv_layers.new(name="UVMap").data.foreach_set("uv", np.array([VT[i] for f in FT for i in f]).ravel())
    for k in ("normal", "ao"):
        tex = path[:-4] + f"_{k}.png"
        if os.path.exists(tex):
            me[f"soma_{k}"] = os.path.abspath(tex)
    me.materials.append(ob.data.materials[0])
    me.materials.append(ob.data.materials[0])                   # vertebrae: plain bone (slot 0 gets the skull maps)
    me.polygons.foreach_set("material_index", np.r_[np.zeros(len(F) - len(neck)), np.ones(len(neck))].astype(np.int32))
    me.shade_smooth()
    ob.data = me


def pose_hands(cache, mode):
    """Fixed finger pose (hand_pose.POSES) for the skeleton's hands: the left hand is curled, the right one is its mirror
    image across the body's mid-plane (x = 0 in the rest frame); the flat hand muscles and the tendon ends in the palm are
    hidden (faces removed)."""
    import bmesh
    from hand_pose import curl_hand
    S = np.load(os.path.join(cache, "static.npz"))
    names = [str(n) for n in S["joint_names"]]
    V = S["bone_rest"].astype(np.float64).copy()
    curl_hand(V, S, "Left", mode)
    jl, jr = names.index("LeftHand"), names.index("RightHand")
    left, right = (bpy.data.objects[f"SOMA_Posed_Bone_{j:02d}"] for j in (jl, jr))
    used = np.where(S["bone_joint"] == jl)[0]                          # the object's vertices, in order (load_cache)
    left.data.vertices.foreach_set("co", V[used].astype(np.float32).ravel())
    left.data.update()
    flip = np.array([-1.0, 1.0, 1.0])
    mirrored = V[used] * flip + S["j_rest"][jr] - S["j_rest"][jl] * flip
    me = bpy.data.meshes.new("SOMA_RightHand")
    me.from_pydata(mirrored.tolist(), [], [tuple(p.vertices)[::-1] for p in left.data.polygons])
    me.materials.append(right.data.materials[0])
    me.shade_smooth()
    right.data = me
    mus = bpy.data.objects["SOMA_Posed_Muscles"].data
    hand = (S["muscle_weights"][:, [jl, jr]] > 0.5).any(1)                # tendon ends that follow the hand into the palm
    for (a, b), n in zip(S["muscle_ranges"], S["muscle_names"]):
        if str(n) in ("l-hand", "r-hand"):
            hand[a:b] = True
    bm = bmesh.new()
    bm.from_mesh(mus)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if all(hand[v.index] for v in f.verts)], context="FACES_ONLY")
    bm.to_mesh(mus)
    bm.free()


def wrist_rotations(cache, roll, bend):
    """Fixed wrist turn (4,4, rest frame, about each wrist joint) per side: `roll` deg about the forearm axis, then `bend`
    deg towards the back of the hand (extension); the right side is the mirror image of the left (as its fingers)."""
    from hand_pose import curl_hand
    S = np.load(os.path.join(cache, "static.npz"))
    names = [str(n) for n in S["joint_names"]]
    along, palm, lateral = curl_hand(S["bone_rest"].astype(np.float64).copy(), S, "Left", "grip", dry=True)["frame"]
    rot = lambda k, a: np.array(Matrix.Rotation(np.radians(a), 3, k.tolist()))
    R = rot(lateral, -bend) @ rot(along, roll)
    M = np.diag([-1.0, 1.0, 1.0])
    out = {}
    for side, Rs in (("Left", R), ("Right", M @ R @ M)):
        p = S["j_rest"][names.index(side + "Hand")]
        G = np.eye(4); G[:3, :3], G[:3, 3] = Rs, p - Rs @ p
        out[side] = G
    return out


def lens_track(cache, T, joint, E, K, W, H, smooth=9):
    """Image position (0..1, origin bottom-left) of a joint per frame, smoothed over `smooth` frames."""
    names = [str(n) for n in np.load(os.path.join(cache, "static.npz"))["joint_names"]]
    j = np.load(os.path.join(cache, "joints_posed.npy"))[:, names.index(joint)]               # (F,3) model frame
    xw = np.einsum("fab,fb->fa", T[:, :3, :3], j) + T[:, :3, 3]
    xc = xw @ E[:3, :3].T + E[:3, 3]
    uv = xc @ K.T
    uv = uv[:, :2] / uv[:, 2:]
    k = np.ones(smooth) / smooth
    uv = np.stack([np.convolve(np.pad(uv[:, d], smooth // 2, mode="edge"), k, mode="valid") for d in (0, 1)], 1)
    return np.c_[uv[:, 0] / W, 1.0 - uv[:, 1] / H]


def head_track(cache, T, E, K, f0, f1, margin, smooth=3):
    """Per scene frame f0..f1: centre and size (px) of the projected face (front half of the skin vertices skinned mostly
    to Head), enlarged by `margin` for the head's pose error (the head has no markers)."""
    S = np.load(os.path.join(cache, "static.npz"))
    names = [str(n) for n in S["joint_names"]]
    idx = np.where(S["skin_weights"].argmax(1) == names.index("Head"))[0]
    fwd = S["j_rest"][names.index("LeftToeBase")] - S["j_rest"][names.index("LeftFoot")]     # rest-pose forward
    fwd[1] = 0
    rest = S["skin_rest"][idx]
    idx = idx[(rest - rest.mean(0)) @ fwd > 0]
    v = np.load(os.path.join(cache, "skin_posed.npy"), mmap_mode="r")[f0 - 1:f1][:, idx].astype(np.float32)
    v = np.einsum("fab,fnb->fna", T[f0 - 1:f1, :3, :3], v) + T[f0 - 1:f1, None, :3, 3]
    xc = v @ E[:3, :3].T + E[:3, 3]
    uv = xc @ K.T
    uv = uv[..., :2] / uv[..., 2:]
    c, ext = (uv.max(1) + uv.min(1)) / 2, (uv.max(1) - uv.min(1)) * margin
    k = np.ones(smooth) / smooth
    sm = lambda x: np.convolve(np.pad(x, smooth // 2, mode="edge"), k, mode="valid")
    return np.stack([sm(c[:, d]) for d in (0, 1)], 1), np.stack([sm(ext[:, d]) for d in (0, 1)], 1)


def face_track(c, ext, boxes, f0, shift=(0, 0), margin=(1.4, 1.7), drop=0.1, window=25, grow=0.03):
    """Blur ellipse from detected face boxes (npz: frames, box (x, y, w, h) video px, NaN = none; `shift` = crop corner)
    where available: the model face (c, ext from head_track, margin 1) gives the motion, the detections its offset and
    size (box x `margin`, centre lowered by `drop` box heights for the beard); detections far from their neighbours are
    dropped, gaps interpolated, and the ellipse grows by `grow` per frame away from a detection (capped at the model
    face x 1.8)."""
    B = np.load(boxes)
    box = np.full((len(c), 4), np.nan)
    rows = np.arange(len(c)) + f0 - 1                                                       # cache frames
    m = (rows >= B["frames"][0]) & (rows <= B["frames"][-1])
    box[m] = B["box"][rows[m] - B["frames"][0]]
    box[:, :2] -= shift
    box[:, 1] += drop * box[:, 3]
    ok = ~np.isnan(box[:, 0])
    off = box[:, :2] + box[:, 2:] / 2 - c
    rel = box[:, 2:] / ext
    for i in np.where(ok)[0]:                                                               # outliers vs the neighbours
        s = ok & (np.abs(np.arange(len(c)) - i) <= window // 2)
        if np.linalg.norm(off[i] - np.median(off[s], 0)) > 0.5 * ext[i].max():
            ok[i] = False
    if not ok.any():
        return c, ext * 1.8
    i_ok, t = np.where(ok)[0], np.arange(len(c))
    off = np.stack([np.interp(t, i_ok, off[i_ok, d]) for d in (0, 1)], 1)
    rel = np.stack([np.interp(t, i_ok, rel[i_ok, d]) for d in (0, 1)], 1)
    dist = np.abs(t[:, None] - i_ok[None]).min(1)
    margin = np.asarray(margin)
    size = np.minimum(rel * ext * margin * (1 + grow * dist[:, None]), np.maximum(rel * ext * margin, ext * 1.8))
    return c + off, size


def legend_scene(sc, text_mat):
    """Scene holding the showcase |SOMA - LBS| colour bar, framed in the upper-left corner by an orthographic camera.
    The compositor lays it over the final image, so the lens does not clip it."""
    lg = bpy.data.scenes.new("SOMA_Legend")
    lg.render.engine, lg.render.film_transparent = sc.render.engine, True
    lg.render.resolution_x, lg.render.resolution_y = sc.render.resolution_x, sc.render.resolution_y
    lg.render.resolution_percentage = sc.render.resolution_percentage
    lg.view_settings.view_transform = "Standard"
    ss._legend("Posed", 0.0, lg.collection, text_mat)                          # bar 0.8 wide centred at x 0, z -0.2
    cam = bpy.data.objects.new("SOMA_LegendCamera", bpy.data.cameras.new("SOMA_LegendCamera"))
    lg.collection.objects.link(cam); lg.camera = cam
    w = 2.8                                                                     # view width: bar = 0.8 / w of the image
    cam.data.type, cam.data.ortho_scale = "ORTHO", w
    aspect = sc.render.resolution_y / sc.render.resolution_x
    cam.location = (w / 2 - 0.2 * w, -5.0, -0.2 - (0.5 - 0.12) * w * aspect)    # bar centre 20 % from the left, 12 % from the top
    cam.rotation_euler = (np.pi / 2, 0, 0)
    return lg


PALETTE = ("#2f8871", "#283477", "#612e65")                                 # retro style: teal, deep blue, purple
NEON_TEAL, NEON_PURPLE = (0.25, 1.0, 0.78), (0.85, 0.35, 1.0)                # linear RGB


def srgb(hex_):
    """Linear RGB (3,) of an sRGB hex colour."""
    c = np.array([int(hex_[i:i + 2], 16) for i in (1, 3, 5)]) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def color_in(mix, ab):
    """Colour input A or B of a Mix node (inputs["A"] is its float socket)."""
    return next(i for i in mix.inputs if i.identifier == f"{ab}_Color")


def set_ramp(cr, stops):
    """Colour ramp stops [(position, RGBA)], ascending. Stops re-sort when moved: the extra ones are added at 0.5
    (between the end stops), so setting ascending positions by index keeps the order."""
    while len(cr.elements) > len(stops):
        cr.elements.remove(cr.elements[-1])
    while len(cr.elements) < len(stops):
        cr.elements.new(0.5)
    for i, (pos, col) in enumerate(stops):
        cr.elements[i].position = pos
    for i, (pos, col) in enumerate(stops):
        cr.elements[i].color = col


def retro_muscles(cache):
    """Retro-futurism muscles: colours from the palette (each muscle's hue walks the teal -> blue -> purple loop, its
    lightness varies so neighbours stay apart) and a neon wireframe."""
    import colorsys
    S = np.load(os.path.join(cache, "static.npz"))
    pal = [colorsys.rgb_to_hsv(*(int(h[i:i + 2], 16) / 255.0 for i in (1, 3, 5))) for h in PALETTE]
    vcol = np.zeros((len(S["muscle_rest"]), 4), np.float32); vcol[:, 3] = 1
    for k, ((a, b), col) in enumerate(zip(S["muscle_ranges"], S["muscle_colors"])):
        h = colorsys.rgb_to_hsv(*col)[0] * 3                                            # position on the palette loop
        p0, p1, t = pal[int(h) % 3], pal[(int(h) + 1) % 3], h - int(h)
        hue = p0[0] + ((p1[0] - p0[0] + 0.5) % 1.0 - 0.5) * t                           # shortest way round the hue circle
        sat = 0.9
        val = (0.55, 0.72, 0.9)[k % 3]                                                 # lighter than the palette itself
        rgb = np.array(colorsys.hsv_to_rgb(hue % 1.0, sat, val))
        vcol[a:b, :3] = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    mus = bpy.data.objects["SOMA_Posed_Muscles"]
    mus.data.color_attributes["muscle_color"].data.foreach_set("color", vcol.ravel())
    m = mus.active_material
    m.node_tree.nodes["WireStrength"].outputs[0].default_value = 0.5
    for n in m.node_tree.nodes:
        if n.type == "EMISSION":                                                        # the wireframe colour
            n.inputs["Color"].default_value = (*NEON_TEAL, 1)
            n.inputs["Strength"].default_value = 0.8


def retro_bones():
    """X-ray bones: own lighting (key light from the upper left through a deep blue -> blue -> soft white ramp),
    ambient occlusion in the crevices and a neon teal rim."""
    b = bpy.data.materials["SOMA_Bone"]                    # camera space: -z = towards the camera
    nt, N, L = b.node_tree, b.node_tree.nodes.new, b.node_tree.links.new
    geo = N("ShaderNodeNewGeometry")
    cam = N("ShaderNodeVectorTransform"); cam.vector_type, cam.convert_from, cam.convert_to = "NORMAL", "WORLD", "CAMERA"
    L(geo.outputs["Normal"], cam.inputs["Vector"])

    key = N("ShaderNodeVectorMath"); key.operation = "DOT_PRODUCT"            # key light from the upper left
    key.inputs[1].default_value = tuple(np.array([-0.45, 0.48, -0.75]) / np.linalg.norm([-0.45, 0.48, -0.75]))
    shade = N("ShaderNodeMapRange"); shade.inputs["From Min"].default_value = -0.3
    ramp = N("ShaderNodeValToRGB")                          # x-ray ramp: deep blue -> blue -> soft white
    set_ramp(ramp.color_ramp, ((0.0, (*(srgb(PALETTE[1]) * 0.6), 1)), (0.45, (0.11, 0.17, 0.58, 1)), (1.0, (0.9, 0.9, 0.94, 1))))
    L(cam.outputs["Vector"], key.inputs[0]); L(key.outputs["Value"], shade.inputs["Value"])
    L(shade.outputs["Result"], ramp.inputs["Fac"])
    ao = N("ShaderNodeAmbientOcclusion"); ao.inputs["Distance"].default_value = 0.015
    occl = N("ShaderNodeMix"); occl.data_type = "RGBA"; occl.blend_type = "MULTIPLY"; occl.inputs["Factor"].default_value = 0.8
    lw = N("ShaderNodeLayerWeight"); lw.inputs["Blend"].default_value = 0.15
    rim = N("ShaderNodeMix"); rim.data_type = "RGBA"; rim.blend_type = "ADD"
    color_in(rim, "B").default_value = (*(np.array(NEON_TEAL) * 0.8), 1)
    em = N("ShaderNodeEmission")
    L(ramp.outputs["Color"], color_in(occl, "A")); L(ao.outputs["Color"], color_in(occl, "B"))
    L(occl.outputs["Result"], color_in(rim, "A")); L(lw.outputs["Facing"], rim.inputs["Factor"])
    L(rim.outputs["Result"], em.inputs["Color"]); L(em.outputs["Emission"], nt.nodes["Material Output"].inputs["Surface"])
    skull = bpy.data.meshes.get("SOMA_Skull")                   # replaced skull: its normal map and baked AO
    if skull is not None and "soma_normal" in skull:
        m = b.copy(); m.name = "SOMA_Skull"
        nt, N, L = m.node_tree, m.node_tree.nodes.new, m.node_tree.links.new
        tex = N("ShaderNodeTexImage"); tex.image = bpy.data.images.load(skull["soma_normal"])
        tex.image.colorspace_settings.name = "Non-Color"
        nmap = N("ShaderNodeNormalMap"); L(tex.outputs["Color"], nmap.inputs["Color"])
        for n in nt.nodes:                                      # every normal input: geometry -> normal map
            for i in n.inputs:
                if i.name == "Normal" or (n.bl_idname == "ShaderNodeVectorTransform" and i.name == "Vector"):
                    L(nmap.outputs["Normal"], i)
        if "soma_ao" in skull:
            ao_tex = N("ShaderNodeTexImage"); ao_tex.image = bpy.data.images.load(skull["soma_ao"])
            ao_tex.image.colorspace_settings.name = "Non-Color"
            occl = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeMix" and n.blend_type == "MULTIPLY")
            both = N("ShaderNodeMix"); both.data_type = "RGBA"; both.blend_type = "MULTIPLY"; both.inputs["Factor"].default_value = 1.0
            L(color_in(occl, "B").links[0].from_socket, color_in(both, "A")); L(ao_tex.outputs["Color"], color_in(both, "B"))
            L(both.outputs[2], color_in(occl, "B"))
        skull.materials[0] = m


def hud_rgba(R, w, t=1.0):
    """RGBA (straight alpha, linear) of the retro lens HUD for a lens of radius R px: neon teal ring, tick marks (major
    every 30 deg) and purple outer arcs; w = line width px. t < 1 = boot-up at progress t: the ring draws on clockwise
    from the top with a bright head, each tick lights as the head passes, the arcs slide in at the end (t = 1: full)."""
    S = int(2 * 1.12 * R) + 8
    y, x = np.mgrid[:S, :S] - (S - 1) / 2
    r, ang = np.hypot(x, y), np.degrees(np.arctan2(y, x)) % 360                       # image y down: clockwise
    ease = lambda u: (lambda c: c * c * (3 - 2 * c))(np.clip(u, 0.0, 1.0))
    line = lambda d, width: np.clip(1 - np.abs(d) / width, 0, 1)                        # anti-aliased stroke
    sweep = 360.0 * ease(t / 0.7)
    behind = (sweep - (ang - 270) % 360) * np.pi / 180 * R                               # arc length behind the head
    drawn = np.clip(behind / w, 0, 1)
    ring = (line(r - R, w) + 0.35 * np.exp(-((r - R) / (4 * w)) ** 2)) * drawn          # stroke + glow
    head = (1 - ease((t - 0.6) / 0.3)) * np.exp(-(behind / (12 * w)) ** 2) * (behind > -w) \
        * np.exp(-((r - R) / (3 * w)) ** 2)                                             # leading glow
    lit = np.clip(behind / (25 * w), 0, 1)                                               # ticks fade in behind the head
    tick = (np.abs(((ang + 2.5) % 5) - 2.5) * np.pi / 180 * r < 0.6 * w) & (r > R * 1.015) & (r < R * 1.035)
    major = (np.abs(((ang + 15) % 30) - 15) * np.pi / 180 * r < 0.9 * w) & (r > R * 1.015) & (r < R * 1.065)
    slide = ease((t - 0.55) / 0.4)                                                       # arcs: rotate into place, fade in
    arcs = line(r - 1.085 * R, 0.6 * w) * (((ang + 20 + 30 * (1 - slide)) % 90) < 50) * slide
    power = 1.0 if t >= 0.12 else 0.5 + 0.5 * np.cos(t * 160) ** 2                       # power-on flicker
    a_teal = np.clip(ring + 1.5 * head + (0.6 * tick + 0.9 * major) * lit, 0, 1) * power
    a_purp = np.clip(arcs, 0, 1) * power
    rgba = np.zeros((S, S, 4), np.float32)
    alpha = np.clip(a_teal + a_purp, 0, 1)
    rgba[..., :3] = (np.array(NEON_TEAL) * a_teal[..., None] + np.array(NEON_PURPLE) * a_purp[..., None]) \
        / np.maximum(alpha, 1e-6)[..., None]
    rgba[..., 3] = alpha
    return rgba[::-1]                                                                    # Blender images start at the bottom


def hud_image(name, R, w, frames=0, folder=None):
    """The HUD as an image: static (frames = 0), or the boot-up as an EXR sequence of `frames` frames in `folder`
    (its last frame is the static HUD, which the sequence holds afterwards)."""
    if not frames:
        rgba = hud_rgba(R, w)
        img = bpy.data.images.new(name, rgba.shape[1], rgba.shape[0], alpha=True, float_buffer=True)
        img.pixels.foreach_set(rgba.ravel())
        img.pack()
        return img
    folder = os.path.abspath(folder)
    os.makedirs(folder, exist_ok=True)
    for i in range(frames):
        rgba = hud_rgba(R, w, i / (frames - 1))
        rgba[..., :3] *= rgba[..., 3:]                                                  # EXR: premultiplied
        img = bpy.data.images.new(f"{name}_tmp", rgba.shape[1], rgba.shape[0], alpha=True, float_buffer=True)
        img.colorspace_settings.name = "Non-Color"                      # saved as is (else sRGB-encoded on save)
        img.pixels.foreach_set(rgba.ravel())
        img.filepath_raw, img.file_format = os.path.join(folder, f"{name}_{i + 1:04d}.exr"), "OPEN_EXR"
        img.save()
        bpy.data.images.remove(img)
    img = bpy.data.images.load(os.path.join(folder, f"{name}_0001.exr"))
    img.source, img.alpha_mode, img.colorspace_settings.name = "SEQUENCE", "PREMUL", "Non-Color"
    return img


def scanline_image(name, w, h, period=3, depth=0.3):
    """Image of horizontal scan lines (1 - depth every `period` rows) to multiply over the x-ray view."""
    row = np.ones(h, np.float32)
    row[::period] = 1 - depth
    rgba = np.ones((h, w, 4), np.float32)
    rgba[..., :3] *= row[:, None, None]
    img = bpy.data.images.new(name, w, h, alpha=True, float_buffer=True)
    img.pixels.foreach_set(rgba.ravel())
    img.pack()
    return img


def compositor(sc, clip, track, radius, W, H, f0, f1, open_frames=0, legend=None, crop=None, face=None, retro=False,
               hud_dir=None, delay=0):
    sc.use_nodes = True
    nt = sc.node_tree
    nt.nodes.clear()
    N, L = nt.nodes.new, nt.links.new
    rl, mc, out = N("CompositorNodeRLayers"), N("CompositorNodeMovieClip"), N("CompositorNodeComposite")
    mc.clip = clip
    src = mc.outputs["Image"]
    if crop is not None:                                                    # x, y (top-left, video px), width, height
        cn = N("CompositorNodeCrop")
        for k, v in zip(("X", "Y", "Width", "Height"), (crop[0], clip.size[1] - crop[1] - crop[3], crop[2], crop[3])):
            cn.inputs[k].default_value = int(v)                             # Y counted from the bottom
        L(src, cn.inputs["Image"]); src = cn.outputs[0]
    video = N("CompositorNodeScale"); video.space = "RENDER_SIZE"; video.frame_method = "STRETCH"
    L(src, video.inputs["Image"])
    vsrc = video.outputs[0]
    if face is not None:                                    # privacy: blur the face in the video (the skull stays sharp)
        c, ext = face
        px = np.median(ext) * sc.render.resolution_percentage / 100                         # face size, render px
        fm = N("CompositorNodeEllipseMask")
        fsoft = N("CompositorNodeBlur"); fsoft.filter_type = "FAST_GAUSS"                    # thin soft edge
        fsoft.inputs["Size"].default_value = (max(1, int(0.06 * px)),) * 2
        L(fm.outputs["Mask"], fsoft.inputs["Image"])
        fblur = N("CompositorNodeBlur"); fblur.filter_type = "GAUSS"
        fblur.inputs["Size"].default_value = (max(2, int(0.35 * px)),) * 2
        L(vsrc, fblur.inputs["Image"])
        fmix = N("CompositorNodeMixRGB")
        L(fsoft.outputs[0], fmix.inputs["Fac"]); L(vsrc, fmix.inputs[1]); L(fblur.outputs[0], fmix.inputs[2])
        vsrc = fmix.outputs[0]
        for f in range(f0, f1 + 1):
            fm.inputs["Position"].default_value = (c[f - f0, 0] / W, 1.0 - c[f - f0, 1] / H)
            fm.inputs["Size"].default_value = tuple(ext[f - f0] / W)                       # both relative to the width
            fm.inputs["Position"].keyframe_insert("default_value", frame=f)
            fm.inputs["Size"].keyframe_insert("default_value", frame=f)
    dark = N("CompositorNodeHueSat")                                                        # x-ray background
    dark.inputs["Saturation"].default_value, dark.inputs["Value"].default_value = (0.35, 0.32) if retro else (0.2, 0.3)
    tint = N("CompositorNodeMixRGB"); tint.blend_type = "MULTIPLY"; tint.inputs["Fac"].default_value = 0.8 if retro else 1.0
    tint.inputs[2].default_value = (*(srgb(PALETTE[1]) * 5.0), 1.0) if retro else (0.55, 0.75, 1.0, 1.0)
    over = N("CompositorNodeAlphaOver")
    L(vsrc, dark.inputs["Image"]); L(dark.outputs[0], tint.inputs[1])
    L(tint.outputs[0], over.inputs[1]); L(rl.outputs["Image"], over.inputs[2])
    xray = over.outputs[0]
    Wr, Hr = (round(v * sc.render.resolution_percentage / 100) for v in (W, H))
    if retro:                                                               # neon bloom + CRT scan lines in the lens
        bloom = N("CompositorNodeGlare"); bloom.glare_type = "BLOOM"; bloom.quality = "HIGH"
        bloom.inputs["Threshold"].default_value, bloom.inputs["Strength"].default_value = 0.95, 0.35
        L(xray, bloom.inputs["Image"])
        scan = N("CompositorNodeImage"); scan.image = scanline_image("SOMA_Scanlines", Wr, Hr)
        smul = N("CompositorNodeMixRGB"); smul.blend_type = "MULTIPLY"; smul.inputs["Fac"].default_value = 1.0
        L(bloom.outputs[0], smul.inputs[1]); L(scan.outputs["Image"], smul.inputs[2])
        xray = smul.outputs[0]
    lens = N("CompositorNodeEllipseMask")
    lens.inputs["Size"].default_value = (2 * radius, 2 * radius)                          # both relative to the width
    edge = int(0.012 * W)
    inner = N("CompositorNodeDilateErode"); inner.mode = "DISTANCE"; inner.inputs["Size"].default_value = -edge
    soft = N("CompositorNodeBlur"); soft.filter_type = "FAST_GAUSS"; soft.size_x = soft.size_y = edge   # soft edge inside the ring
    L(lens.outputs["Mask"], inner.inputs["Mask"]); L(inner.outputs[0], soft.inputs["Image"])
    reveal = N("CompositorNodeMixRGB")
    L(soft.outputs[0], reveal.inputs["Fac"]); L(vsrc, reveal.inputs[1]); L(xray, reveal.inputs[2])
    grow, shrink = N("CompositorNodeDilateErode"), N("CompositorNodeDilateErode")
    for n, s in ((grow, 1), (shrink, -1)):
        n.mode = "DISTANCE"; n.inputs["Size"].default_value = s * max(2, int(0.0012 * W))
        L(lens.outputs["Mask"], n.inputs["Mask"])
    ring = N("CompositorNodeMath"); ring.operation = "SUBTRACT"
    L(grow.outputs[0], ring.inputs[0]); L(shrink.outputs[0], ring.inputs[1])
    glow = N("CompositorNodeMixRGB"); glow.blend_type = "ADD"; glow.inputs[2].default_value = (0.6, 0.85, 1.0, 1.0)
    L(ring.outputs[0], glow.inputs["Fac"]); L(reveal.outputs[0], glow.inputs[1])
    final = glow
    if retro:                                                               # HUD ring instead of the plain ring
        hud = N("CompositorNodeImage")
        hud.image = hud_image("SOMA_HUD", radius * Wr, max(1.5, 0.0012 * Wr), open_frames, hud_dir)
        if open_frames:                                                     # boot-up sequence, then held
            hud.frame_duration, hud.frame_start, hud.use_cyclic = open_frames, f0 + delay, False
        hmove = N("CompositorNodeTranslate")
        hover = N("CompositorNodeAlphaOver")
        L(hud.outputs["Image"], hmove.inputs["Image"])
        L(reveal.outputs[0], hover.inputs[1]); L(hmove.outputs[0], hover.inputs[2])
        for f, v in ((f0 + delay - 1, 0.0), (f0 + delay, 1.0)) if delay else ():    # no HUD before the boot-up
            hover.inputs["Fac"].default_value = v
            hover.inputs["Fac"].keyframe_insert("default_value", frame=f)
        fringe = N("CompositorNodeLensdist"); fringe.inputs["Dispersion"].default_value = 0.012   # chromatic fringe
        L(hover.outputs[0], fringe.inputs["Image"])
        final = fringe
        for f in range(f0, f1 + 1):
            hmove.inputs["X"].default_value = (track[f - 1][0] - 0.5) * Wr
            hmove.inputs["Y"].default_value = (track[f - 1][1] - 0.5) * Hr
            for k in ("X", "Y"):
                hmove.inputs[k].keyframe_insert("default_value", frame=f)
    if legend is not None:                                                                  # colour bar on top, unclipped
        lrl = N("CompositorNodeRLayers"); lrl.scene = legend
        over_lg = N("CompositorNodeAlphaOver")
        L(final.outputs[0], over_lg.inputs[1]); L(lrl.outputs["Image"], over_lg.inputs[2])
        final = over_lg
    L(final.outputs[0], out.inputs["Image"])
    viewer = N("CompositorNodeViewer"); L(final.outputs[0], viewer.inputs["Image"])
    pos = lens.inputs["Position"]
    for f in range(f0, f1 + 1):                                                             # scene frame f = cache frame f-1
        pos.default_value = track[f - 1]
        pos.keyframe_insert("default_value", frame=f)
    if open_frames:                                                     # grow-open reveal: closed lens -> full radius
        size = lens.inputs["Size"]                                          # (retro: from the centre, while the HUD boots)
        s = f0 + delay
        keys = ((s, 0.0), (s + round(0.15 * open_frames), 0.0), (s + round(0.85 * open_frames), radius)) \
            if retro else ((s, 0.0), (s + open_frames, radius))
        for f, r in keys:
            size.default_value = (2 * r, 2 * r)
            size.keyframe_insert("default_value", frame=f)


def build(args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ss.register()
    sc = bpy.context.scene
    engines = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
    sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
    sc.view_settings.view_transform = "Standard"
    sc.render.film_transparent = True
    sc.sync_mode = "FRAME_DROP"
    world = bpy.data.worlds.new("SOMA_World"); sc.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.35, 0.35, 0.35, 1)

    # Characters: only the posed stage, placed in the capture world
    ss._muscle_material("SOMA_Muscles_Posed")
    ss._simple_material("SOMA_Bone", (0.85, 0.83, 0.78))
    ss.STAGES.clear()
    ss.STAGES["Posed"] = (0.0, True)
    sc.soma_show_skin = False
    ss.load_cache(sc, args.cache)
    if args.skull_mesh:
        replace_skull(args.cache, args.skull_mesh)
    if args.hands:
        pose_hands(args.cache, args.hands)
    T = np.load(os.path.join(args.cache, "world_T.npy"))
    f0, f1 = (args.frames[0] + 1, args.frames[1] + 1) if args.frames else (1, len(T))
    D = None
    if args.markers:
        ids = np.load(os.path.join(args.cache, "markers_world.npz"))["ids"]
        Q = triangulation(args.markers, ids, f0 - 1, f1 - 1)
        g0, g1 = args.fit_frames or (f0 - 1, f1 - 1)                    # same range -> renders that line up exactly
        C = capture_correction(args.cache, triangulation(args.markers, ids, g0, g1) if args.fit_frames else Q,
                               args.detections, g0, g1)
        if args.bone_fit:
            D = bone_correction(args.cache, T, C, Q, f0 - 1, f1 - 1, max_angle=args.bone_max_angle)
        T = C @ T
    sc.frame_start, sc.frame_end = f0, f1

    # Calibrated camera + video
    clip = bpy.data.movieclips.load(os.path.abspath(args.video))
    clip.frame_start = 1                                                                    # video frame i = cache frame i
    W, H = clip.size
    E, K = read_camera(args.calib, args.camera, W)
    K[:2, 2] += args.offset                                                                 # manual 2D nudge of the overlay
    if args.crop:                                                           # render only this part of the video
        K[:2, 2] -= args.crop[:2]
        W, H = args.crop[2:]
    Tc = T
    if args.depth:                                  # manual depth nudge: along the line of sight to the body (scale only)
        T = T.copy()
        c = np.einsum("fab,fb->fa", T[:, :3, :3], np.load(os.path.join(args.cache, "joints_posed.npy")).mean(1)) + T[:, :3, 3]
        ray = c + E[:3, :3].T @ E[:3, 3]                                                   # camera centre -> body centre
        T[:, :3, 3] += args.depth * ray / np.linalg.norm(ray, axis=1, keepdims=True)
    root = bpy.data.objects["SOMA_Posed"]
    root.rotation_mode = "QUATERNION"
    for f in range(f0, f1 + 1):
        root.matrix_world = Matrix((Y_UP_TO_Z_UP @ T[f - 1]).tolist())
        root.keyframe_insert("location", frame=f)
        root.keyframe_insert("rotation_quaternion", frame=f)
    names = [str(n) for n in np.load(os.path.join(args.cache, "static.npz"))["joint_names"]]

    def placed(joint, base=None):                                   # body placement incl. that joint's correction
        base = T if base is None else base
        if D is None:
            return base
        Tj = base.copy()
        Tj[f0 - 1:f1] = base[f0 - 1:f1] @ np.linalg.inv(Tc[f0 - 1:f1]) @ D[:, names.index(joint)] @ Tc[f0 - 1:f1]
        return Tj
    if args.face_landmarks:                         # neck + head towards the detected face (--depth: Sd = T Tc^-1)
        if D is None:
            D = np.tile(np.eye(4), (f1 - f0 + 1, len(names), 1, 1))
        G = face_correction(args.cache, placed("Head"), args.face_landmarks, E, K, f0, f1,
                            args.crop[:2] if args.crop else (0, 0))
        Sd = T[f0 - 1:f1] @ np.linalg.inv(Tc[f0 - 1:f1])
        for j in ("Neck", "Head"):
            D[:, names.index(j)] = np.linalg.inv(Sd) @ G @ Sd @ D[:, names.index(j)]
    if args.hands or args.head_neutral:
        if D is None:
            D = np.tile(np.eye(4), (f1 - f0 + 1, len(names), 1, 1))
        A = np.tile(np.eye(4), (f1 - f0 + 1, len(names), 1, 1))
        A[..., :3, :] = np.load(os.path.join(args.cache, "A.npy"))[f0 - 1:f1]              # skinning transforms (model)
        Tr = Tc[f0 - 1:f1]
    if args.head_neutral:                           # neck + head rigidly follow the upper spine (rest angle)
        s4 = names.index("Spine4")
        for j in ("Neck", "Head"):
            D[:, names.index(j)] = D[:, s4] @ Tr @ A[:, s4] @ np.linalg.inv(A[:, names.index(j)]) @ np.linalg.inv(Tr)
    if args.hands:                                  # straight wrist: each hand rigidly follows its forearm (rest angle)
        Rw = wrist_rotations(args.cache, args.wrist_roll, args.wrist_bend)
        for side in ("Left", "Right"):
            h, fa = names.index(side + "Hand"), names.index(side + "ForeArm")
            Dm = A[:, fa] @ Rw[side] @ np.linalg.inv(A[:, h])                         # hand posed as the forearm
            D[:, h] = D[:, fa] @ Tr @ Dm @ np.linalg.inv(Tr)
    if D is not None:                                               # corrections in the model frame: Tc^-1 D Tc
        Tr = Tc[f0 - 1:f1]
        bone_rig(root, args.cache, np.linalg.inv(Tr)[:, None] @ D @ Tr[:, None], f0)
    if args.hide_skull:
        skull = bpy.data.objects[f"SOMA_Posed_Bone_{names.index('Head'):02d}"]
        skull.hide_render = skull.hide_viewport = True
    cam = bpy.data.objects.new("SOMA_Camera", bpy.data.cameras.new("SOMA_Camera"))
    sc.collection.objects.link(cam); sc.camera = cam
    cam.matrix_world = Matrix((Y_UP_TO_Z_UP @ np.linalg.inv(E) @ CV_TO_BLENDER_CAM).tolist())
    cd = cam.data
    cd.sensor_fit, cd.sensor_width = "HORIZONTAL", 36.0
    cd.lens = K[0, 0] / W * cd.sensor_width
    cd.shift_x, cd.shift_y = (W / 2 - K[0, 2]) / W, (K[1, 2] - H / 2) / W
    cd.clip_start, cd.clip_end = 0.1, 100.0
    cd.show_background_images = True
    bg = cd.background_images.new(); bg.source, bg.clip = "MOVIE_CLIP", clip
    sc.render.resolution_x, sc.render.resolution_y = W, H
    sc.render.resolution_percentage = int(args.scale * 100)
    sun = bpy.data.objects.new("SOMA_Sun", bpy.data.lights.new("SOMA_Sun", "SUN"))
    sun.data.energy = 3.0
    sc.collection.objects.link(sun)
    sun.matrix_world = cam.matrix_world                                                     # light from the camera

    if args.style == "retro":
        if args.color == "muscle":                                          # the heatmap keeps its colours and wireframe
            retro_muscles(args.cache)
        retro_bones()
    legend = None
    if args.color == "heatmap":
        legend = legend_scene(sc, ss._simple_material("SOMA_Text", (0.92, 0.92, 0.92), emission=1.0))
        sc.soma_disp_max = args.disp_max
        sc.soma_color_posed = "DISP"                                        # also shows the legend and sets its max label
    sc.soma_scale = args.soma_scale
    face = None
    if args.blur_face:                              # the real head, in the video
        face = head_track(args.cache, placed("Head", Tc), E, K, f0, f1, 1.0 if args.face_boxes else args.blur_margin)
        if args.face_boxes:
            face = face_track(*face, args.face_boxes, f0, args.crop[:2] if args.crop else (0, 0))
    compositor(sc, clip, lens_track(args.cache, placed(args.lens_joint), args.lens_joint, E, K, W, H), args.lens_radius,
               W, H, f0, f1, round(args.lens_open * sc.render.fps), None if args.no_legend else legend, args.crop, face,
               args.style == "retro",
               os.path.join(args.cache, "hud"), round(args.lens_delay * sc.render.fps))
    sc.frame_set(f0)
    txt = bpy.data.texts.load(ss.__file__)                                                  # the player, run on load
    txt.use_module = True
    if args.save:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save))
        print(f"[XRAY] saved {args.save}")
    if args.render:
        r = sc.render
        r.filepath = os.path.abspath(args.render)
        r.image_settings.file_format = "FFMPEG"
        r.ffmpeg.format, r.ffmpeg.codec, r.ffmpeg.constant_rate_factor = "MPEG4", "H264", "HIGH"
        bpy.ops.render.render(animation=True)
        print(f"[XRAY] rendered {args.render}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--video", required=True, help="capture video of the shot (streamNNN.mp4)")
    ap.add_argument("--calib", required=True, help="cameras.calib of the shot")
    ap.add_argument("--camera", type=int, required=True, help="camera index of the video (streamNNN -> NNN)")
    ap.add_argument("--markers", help="raw triangulated markers of the shot (*_transformed_filtered.json)")
    ap.add_argument("--detections", help="2D marker detections of this camera (corners_markers_streamNNN.json)")
    ap.add_argument("--offset", type=float, nargs=2, default=(0.0, 0.0), metavar=("DX", "DY"),
                    help="shift the overlay (and lens) by DX, DY video pixels (+x right, +y down)")
    ap.add_argument("--bone-fit", action="store_true",
                    help="with --markers: also correct each bone onto the triangulated markers (display registration)")
    ap.add_argument("--bone-max-angle", type=float, default=30.0,
                    help="with --bone-fit: a bone whose fit turns more than this (deg) from its parent's takes the parent's")
    ap.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                    help="render only this part of the video (top-left corner and size, video px)")
    ap.add_argument("--blur-face", action="store_true", help="blur the face in the video for privacy (the skull stays sharp)")
    ap.add_argument("--blur-margin", type=float, default=1.6,
                    help="blurred area relative to the projected head (the head has no markers: pose error)")
    ap.add_argument("--face-boxes", help="with --blur-face: detected face boxes per frame (npz: frames, box x y w h in "
                                         "video px, NaN = none) to place and size the blur")
    ap.add_argument("--hide-skull", action="store_true", help="do not show the skull (x-ray starts at the neck)")
    ap.add_argument("--head-neutral", action="store_true",
                    help="neck and head follow the upper spine at their rest angle (when the face cannot be detected)")
    ap.add_argument("--skull-mesh", help="replace the skull by this OBJ (model rest frame of the subject)")
    ap.add_argument("--hands", choices=("grip", "relaxed"),
                    help="fixed finger pose and a straight wrist (the capture has no finger data and unreliable wrist angles)")
    ap.add_argument("--wrist-roll", type=float, default=0.0,
                    help="with --hands: turn both hands about the forearm axis (deg, mirrored on the right)")
    ap.add_argument("--wrist-bend", type=float, default=0.0,
                    help="with --hands: bend both wrists back (extension, deg)")
    ap.add_argument("--face-landmarks", help="face landmarks per frame (npz: frames, lm (F, 5, 2): right eye, left eye, "
                                             "nose tip, right / left mouth corner, video px, NaN = none) to align the head")
    ap.add_argument("--depth", type=float, default=0.0,
                    help="move the body along the line of sight by this many metres (+ away from the camera: smaller)")
    ap.add_argument("--frames", type=int, nargs=2, help="first and last cache frame (default: all)")
    ap.add_argument("--fit-frames", type=int, nargs=2, metavar=("A", "B"),
                    help="with --markers: fit the capture correction on cache frames A..B instead of --frames (renders of "
                         "different ranges of a shot then line up exactly)")
    ap.add_argument("--lens-joint", default="Spine2", help="joint the lens follows")
    ap.add_argument("--lens-radius", type=float, default=0.2, help="lens radius, fraction of the image width")
    ap.add_argument("--lens-open", type=float, default=1.0, help="seconds the lens takes to open at the start (0: open)")
    ap.add_argument("--lens-delay", type=float, default=0.0, help="seconds of plain video before the lens opens")
    ap.add_argument("--color", choices=("muscle", "heatmap"), default="muscle",
                    help="one colour per muscle, or |SOMA - LBS| per vertex with a colour bar")
    ap.add_argument("--disp-max", type=float, default=10.0, help="heatmap colour-bar maximum (mm)")
    ap.add_argument("--style", choices=("default", "retro"), default="default",
                    help="retro: palette muscles, neon wireframe and bone rims, blue-tinted scan-lined lens with a HUD ring")
    ap.add_argument("--soma-scale", type=float, default=1.0,
                    help="deformation scale: 1 = SOMA, 0 = the LBS baseline (with --color heatmap: uniformly 0)")
    ap.add_argument("--no-legend", action="store_true", help="with --color heatmap: no colour bar (to add it later)")
    ap.add_argument("--scale", type=float, default=0.5, help="render resolution relative to the video")
    ap.add_argument("--save", help="save the scene as .blend")
    ap.add_argument("--render", help="render the frame range to this .mp4")
    build(ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []))
