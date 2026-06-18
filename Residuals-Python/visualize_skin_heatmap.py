"""
visualize_skin_heatmap.py -- smooth surface heatmap of the marker residual on the canonical skin mesh.

Per-marker the residual is reduced to a SCALAR (so there is no per-marker un-pose seam at bone
boundaries), interpolated onto the skin vertices, and Laplacian-smoothed over the mesh connectivity
(surface-aware, like the old two-pass refinement). The skin is coloured by that scalar and optionally
displaced along its surface normal.

  Metric "Magnitude"      : |observed - LBS| in world space (frame-independent, fully seam-free).
  Metric "Signed normal"  : (un-posed residual) . canonical surface normal -- the in/out bulge,
                            matching the old 'pure deformation' signed-distance maps.

Run (default S1/shot_001):
  /CT/SOMA/static00/miniforge3/envs/soma/bin/python visualize_skin_heatmap.py [--subject S1] [--shot shot_001] [--port 8081]
"""
import glob
import time
import argparse
import numpy as np
import trimesh
import scipy.sparse as sp
from scipy.spatial import cKDTree
import matplotlib

from visualize_residuals_viser import Shot, rot_x, kabsch, scan

DATASET = "/CT/SOMA/static00/SKIM_dataset"
STATIC = "/CT/SOMA/static00"
K = 8
CLAMP_M = 0.25          # hard ceiling on a marker's scalar (m); catches unpose blow-ups
MAD_K = 6.0             # robust outlier threshold (modified z-score) for the per-frame safety net
FROZEN_TOK = ("head", "hand", "foot", "toe")   # joints whose region is never displaced
MAGNITUDE, SIGNED = "Magnitude", "Signed normal (bulge)"


def skin_mesh_path(subject):
    for pat in (f"{STATIC}/{subject}/layers/tpose/skin_layer-{subject}-TPose.obj",
                f"{STATIC}/{subject}/layers/tpose/skin_layer-*-TPose.obj",
                f"{STATIC}/{subject}/layers/tpose/skin_layer*TPose*.obj"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no canonical skin mesh for {subject}")

def muscle_mesh_path(subject):
    for pat in (f"{STATIC}/{subject}/layers/tpose/musc_layer-{subject}-TPose.obj",
                f"{STATIC}/{subject}/layers/tpose/musc_layer-*-TPose.obj",
                f"{STATIC}/{subject}/layers/tpose/musc_layer*TPose*.obj"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no canonical muscle mesh for {subject}")


def colormap_rgb(values, vmax, signed):
    """values, vmax in cm. signed -> diverging [-vmax,vmax]; else viridis [0,vmax]."""
    cmap = matplotlib.colormaps["viridis"]
    if signed:
        t = np.clip(values / (2 * max(vmax, 1e-6)) + 0.5, 0.0, 1.0)
    else:
        t = np.clip(values / max(vmax, 1e-6), 0.0, 1.0)
    return (cmap(t)[:, :3] * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S1")
    ap.add_argument("--shot", default="shot_001")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    subs, shots = scan(DATASET)
    if not subs:
        raise SystemExit(f"no subjects found under {DATASET}")

    # ── mutable state: updated by load_subject / load_shot ────────────
    state = {"_sc_cache": {}, "_sc_key": [None]}

    def load_subject(subject):
        mesh = trimesh.load(muscle_mesh_path(subject), process=False)
        raw = (rot_x(90.0) @ np.asarray(mesh.vertices).T).T
        verts, inv = np.unique(np.round(raw, 5), axis=0, return_inverse=True)
        faces = inv[np.asarray(mesh.faces)]
        V = len(verts)
        cmesh = trimesh.Trimesh(verts, faces, process=False)
        vnorm = np.asarray(cmesh.vertex_normals, dtype=np.float64)
        E = cmesh.edges_unique
        rows = np.concatenate([E[:, 0], E[:, 1]]); cols = np.concatenate([E[:, 1], E[:, 0]])
        A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(V, V)).tocsr()
        deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1.0
        Anorm = sp.diags(1.0 / deg) @ A
        state.update({"verts": verts, "faces": faces, "V": V, "vnorm": vnorm,
                      "Anorm": Anorm, "_raw_nv": len(mesh.vertices)})

    def load_shot(subject, shot):
        sh = Shot(DATASET, subject, shot)
        verts = state["verts"]; vnorm = state["vnorm"]
        Anorm = state["Anorm"]; V = state["V"]
        # frozen/disp_mask (depends on sh.j_rest / parents)
        cz = np.load(f"{DATASET}/{subject}/canonical.npz", allow_pickle=True)
        jnames = [str(n).lower() for n in cz["joint_names"]]
        joints_z = (sh.ROUT @ sh.j_rest.T).T
        _, v2j = cKDTree(joints_z).query(verts)
        excl = np.array([any(t in n for t in FROZEN_TOK) for n in jnames])
        frozen = np.zeros(V, dtype=bool)
        for j in np.where(excl)[0]:
            sel = v2j == j; p = sh.parents[j]
            if p < 0:
                frozen |= sel; continue
            axis = joints_z[j] - joints_z[p]
            axis = axis / (np.linalg.norm(axis) + 1e-9)
            frozen |= sel & ((verts - joints_z[j]) @ axis > 0.0)
        disp_mask = (~frozen).astype(np.float64)
        for _ in range(15):
            disp_mask = 0.5 * disp_mask + 0.5 * (Anorm @ disp_mask)
        disp_mask = np.clip(disp_mask, 0.0, 1.0)
        # marker interpolation weights
        dist, idx = cKDTree(sh.p_canon).query(verts, k=K)
        iw = 1.0 / (dist ** 2 + 1e-8)
        _, m2v = cKDTree(verts).query(sh.p_canon)
        mnorm = vnorm[m2v]
        bones = np.array([(i, sh.parents[i]) for i in range(len(sh.parents)) if sh.parents[i] >= 0])
        skel_root = int(np.where(sh.parents < 0)[0][0])
        print(f"[skin] {subject}/{shot}: {state['_raw_nv']}v -> {V} welded, "
              f"{sh.M} markers, {sh.F} frames, {int(frozen.sum())} frozen "
              f"({sorted({n for n, e in zip(jnames, excl) if e})})")
        print("[skin] precomputing joint positions ...", end=" ", flush=True)
        jw_all = np.stack([sh.joints_world(f) for f in range(sh.F)])
        print("done")
        state.update({"sh": sh, "disp_mask": disp_mask, "iw": iw, "idx": idx,
                      "mnorm": mnorm, "bones": bones, "skel_root": skel_root, "jw_all": jw_all})
        state["_sc_cache"].clear()
        state["_sc_key"][0] = None

    load_subject(args.subject)
    load_shot(args.subject, args.shot)

    import viser
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    g = server.gui

    init_sub  = args.subject if args.subject in subs else subs[0]
    init_shot = args.shot if args.shot in shots[init_sub] else shots[init_sub][0]
    d_sub    = g.add_dropdown("Subject", options=tuple(subs), initial_value=init_sub)
    d_shot   = g.add_dropdown("Shot", options=tuple(shots[init_sub]), initial_value=init_shot)
    d_metric = g.add_dropdown("Metric", options=(MAGNITUDE, SIGNED), initial_value=SIGNED)
    s_frame  = g.add_slider("Frame", min=0, max=state["sh"].F - 1, step=1, initial_value=0)
    c_play   = g.add_checkbox("Play", initial_value=False)
    s_speed  = g.add_slider("Speed (fps)", min=1, max=60, step=1, initial_value=40)
    s_iter   = g.add_slider("Laplacian smoothing", min=0, max=80, step=1, initial_value=20)
    s_vmax   = g.add_slider("Heatmap max (cm)", min=0.5, max=15.0, step=0.5, initial_value=8.0)
    s_disp   = g.add_slider("Surface displacement gain", min=0.0, max=5.0, step=0.05, initial_value=0.1)
    c_robust = g.add_checkbox("Reject outliers (safety net)", initial_value=True)
    c_freeze = g.add_checkbox("Freeze head/hands/feet", initial_value=True)
    c_global = g.add_checkbox("Remove global (pure deformation)", initial_value=True)
    c_wire   = g.add_checkbox("Show wireframe", initial_value=False)
    s_wire_opacity = g.add_slider("Wireframe opacity", min=0.0, max=1.0, step=0.05, initial_value=0.3)
    c_skel   = g.add_checkbox("Show BVH skeleton", initial_value=True)
    s_skel_rot = g.add_slider("Skeleton rot Z (deg)", min=-180, max=180, step=1, initial_value=90)
    s_skel_dx  = g.add_slider("Skeleton offset X (m)", min=-3.0, max=3.0, step=0.01, initial_value=1.5)
    s_skel_dy  = g.add_slider("Skeleton offset Y (m)", min=-3.0, max=3.0, step=0.01, initial_value=0.0)
    s_skel_dz  = g.add_slider("Skeleton offset Z (m)", min=-3.0, max=3.0, step=0.01, initial_value=0.9)
    txt = g.add_text("Info", initial_value="", disabled=True)

    def scalar_field(f):
        sh = state["sh"]; iw = state["iw"]; idx = state["idx"]
        mnorm = state["mnorm"]; Anorm = state["Anorm"]
        r = sh.res[f]
        if c_global.value:
            lbs = sh.lbs_world(f); obs = lbs + r; fit = sh.mask[f]
            if fit.sum() >= 3:
                s, R, t = kabsch(lbs[fit], obs[fit], False)
                r = obs - ((s * (R @ lbs.T).T) + t)
        if d_metric.value == SIGNED:
            dC = np.nan_to_num(sh.unpose(f, r))
            val_m = np.einsum('mc,mc->m', dC, mnorm)
        else:
            val_m = np.linalg.norm(r, axis=1)
        val_m = np.nan_to_num(val_m, nan=0.0, posinf=0.0, neginf=0.0)
        mvis = sh.mask[f].astype(bool).copy()
        if c_robust.value and mvis.sum() >= 8:
            v = val_m[mvis]; med = np.median(v)
            mad = np.median(np.abs(v - med)) + 1e-9
            mvis &= np.abs(val_m - med) <= MAD_K * 1.4826 * mad
        np.clip(val_m, -CLAMP_M, CLAMP_M, out=val_m)
        mvis = mvis.astype(np.float64)
        we = iw * mvis[idx]; ws = we.sum(1, keepdims=True)
        we = np.divide(we, ws, out=np.zeros_like(we), where=ws > 1e-9)
        val_v = (we * val_m[idx]).sum(1)
        for _ in range(int(s_iter.value)):
            val_v = 0.5 * val_v + 0.5 * (Anorm @ val_v)
        return val_v

    _rz = {"deg": None, "mat": None}

    def get_rot_z():
        d = s_skel_rot.value
        if _rz["deg"] != d:
            c, sv = np.cos(np.deg2rad(d)), np.sin(np.deg2rad(d))
            _rz["mat"] = np.array([[c, -sv, 0.], [sv, c, 0.], [0., 0., 1.]])
            _rz["deg"] = d
        return _rz["mat"]

    def draw_skeleton(f):
        if not c_skel.value:
            server.scene.add_line_segments("/skeleton", points=np.zeros((1, 2, 3), np.float32),
                                           colors=np.zeros((1, 2, 3), np.uint8), line_width=1e-3)
            server.scene.add_point_cloud("/joints", points=np.zeros((1, 3), np.float32),
                                         colors=np.zeros((1, 3), np.uint8), point_size=1e-4)
            return
        jw = state["jw_all"][f].copy()
        jw -= jw[state["skel_root"]]
        jw = (get_rot_z() @ jw.T).T
        jw = jw + np.array([s_skel_dx.value, s_skel_dy.value, s_skel_dz.value])
        seg = jw[state["bones"]].astype(np.float32)
        server.scene.add_line_segments("/skeleton", points=seg,
                                       colors=np.broadcast_to((255, 80, 80), seg.shape).astype(np.uint8),
                                       line_width=3.0)
        server.scene.add_point_cloud("/joints", points=jw.astype(np.float32),
                                     colors=np.broadcast_to((255, 200, 0), jw.shape).astype(np.uint8),
                                     point_size=0.02, point_shape="circle")

    def cached_scalar(f):
        key = (d_metric.value, int(s_iter.value), bool(c_robust.value), bool(c_global.value))
        sc = state["_sc_cache"]; sk = state["_sc_key"]
        if sk[0] != key:
            sc.clear(); sk[0] = key
        if f not in sc:
            sc[f] = scalar_field(f)
        return sc[f]

    def render(_=None):
        sh = state["sh"]; verts = state["verts"]; faces = state["faces"]
        vnorm = state["vnorm"]; disp_mask = state["disp_mask"]
        f = int(s_frame.value)
        val = np.nan_to_num(cached_scalar(f))
        signed = d_metric.value == SIGNED
        colors = colormap_rgb(val * 100.0, s_vmax.value, signed)
        dm = disp_mask if c_freeze.value else 1.0
        nv = np.ascontiguousarray(verts + s_disp.value * (val * dm)[:, None] * vnorm, dtype=np.float32)
        if not np.isfinite(nv).all():
            nv = verts.astype(np.float32)
        m = trimesh.Trimesh(vertices=nv, faces=faces, vertex_colors=colors, process=False)
        server.scene.add_mesh_trimesh("/skin", mesh=m)
        if c_wire.value:
            nv_wire = nv + 0.002 * vnorm.astype(np.float32)
            server.scene.add_mesh_simple("/wireframe", vertices=nv_wire, faces=faces.astype(np.uint32),
                                         color=(0, 0, 0), wireframe=True,
                                         opacity=float(s_wire_opacity.value))
        else:
            server.scene.add_mesh_simple("/wireframe", vertices=np.zeros((3, 3), np.float32),
                                         faces=np.array([[0, 1, 2]], np.uint32),
                                         color=(0, 0, 0), wireframe=True, opacity=0.0)
        draw_skeleton(f)
        lo, hi = val.min() * 100, val.max() * 100
        txt.value = (f"frame {f}: {int(sh.mask[f].sum())} markers obs | "
                     f"{d_metric.value}: [{lo:.1f}, {hi:.1f}] cm | scale +/-{s_vmax.value:.1f} cm")

    ui = {"dirty": True}

    def mark_dirty(_=None):
        ui["dirty"] = True

    @d_sub.on_update
    def _(_):
        load_subject(d_sub.value)
        d_shot.options = tuple(shots[d_sub.value])
        d_shot.value = shots[d_sub.value][0]          # triggers d_shot.on_update

    @d_shot.on_update
    def _(_):
        load_shot(d_sub.value, d_shot.value)
        s_frame.max = state["sh"].F - 1
        s_frame.value = 0
        _rz["deg"] = None                             # invalidate rot_z cache
        ui["dirty"] = True

    for w in (d_metric, s_frame, s_iter, s_vmax, s_disp, c_robust, c_freeze, c_global,
              c_wire, s_wire_opacity, c_skel, s_skel_rot, s_skel_dx, s_skel_dy, s_skel_dz):
        w.on_update(mark_dirty)

    if args.smoke:
        for met in (MAGNITUDE, SIGNED):
            d_metric.value = met; render()
        print("[skin] smoke build OK"); return

    render()
    print(f"[skin] open http://localhost:{args.port}  (Ctrl+C to quit)")
    while True:
        if c_play.value:
            s_frame.value = (int(s_frame.value) + 1) % state["sh"].F
            render(); time.sleep(1.0 / s_speed.value)
        elif ui["dirty"]:
            ui["dirty"] = False; render(); time.sleep(1.0 / 30)
        else:
            time.sleep(0.03)


if __name__ == "__main__":
    main()
