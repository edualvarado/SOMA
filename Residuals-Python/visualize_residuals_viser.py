"""
visualize_residuals_viser.py -- browse the SKIM residual dataset in Viser.

Pick a SUBJECT and SHOT from dropdowns; the viewer reconstructs the LBS markers from the
stored pose and shows the soft-tissue residual in two spaces:

  * World (posed):  LBS cloud (grey) + (LBS + gain*residual) coloured by |residual|.
                    gain 1 reproduces the observed markers.
  * Canonical (T-pose):  the residual un-posed onto the static canonical model -- the body is
                    frozen and you watch the surface points wobble in place.

"Remove global" (Off / Rigid / Rigid+scale) per-frame Procrustes-aligns LBS->observed and shows
only the leftover local field. "Display gain" scales the shown displacement (the real residual,
~5 cm, exceeds the ~2-3 cm marker spacing, so ~0.1-0.3 reads cleanly on the dense cloud).

Run:
  /CT/SOMA/static00/miniforge3/envs/soma/bin/python visualize_residuals_viser.py [--dataset DIR] [--port 8080]
Then open http://localhost:8080  (SSH tunnel if remote:  ssh -L 8080:localhost:8080 <host>).
"""
import os
import glob
import time
import argparse
import numpy as np

from batch_residuals import lbs_frame, rot_x, DEGREE_BVH_X   # reuse the exact LBS + conventions

DATASET_DEFAULT = "/CT/SOMA/static00/SKIM_dataset"
WORLD, CANON = "World (posed)", "Canonical (T-pose)"
G_OFF, G_RIGID, G_SCALE = "Off", "Rigid", "Rigid+scale"


def rotmat_from_6d(m6):
    """Invert batch_residuals.rotmat_to_6d, which stored R[:, :2] row-major flattened, i.e.
    m6 = [R00,R01,R10,R11,R20,R21]. So reshape to (...,3,2) gives the first two COLUMNS."""
    m = m6.reshape(m6.shape[:-1] + (3, 2))
    a1, a2 = m[..., 0], m[..., 1]                       # columns 0 and 1
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)              # columns -> (...,3,3)


def kabsch(P, Q, with_scale):
    Pc, Qc = P.mean(0), Q.mean(0)
    P0, Q0 = P - Pc, Q - Qc
    U, S, Vt = np.linalg.svd(P0.T @ Q0)
    dd = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, dd]) @ U.T
    s = (S * np.array([1, 1, dd])).sum() / ((P0 ** 2).sum() + 1e-12) if with_scale else 1.0
    return s, R, Qc - s * (R @ Pc)


def turbo_colors(values_cm, vmax_cm):
    import matplotlib
    cmap = matplotlib.colormaps["turbo"]
    t = np.clip(values_cm / max(vmax_cm, 1e-6), 0.0, 1.0)
    return (cmap(t)[:, :3] * 255).astype(np.uint8)


class Shot:
    """One shot: reconstructs LBS world markers per frame from the stored pose."""

    def __init__(self, dataset, subject, shot):
        d = np.load(f"{dataset}/{subject}/{shot}.npz", allow_pickle=True)
        c = np.load(f"{dataset}/{subject}/canonical.npz", allow_pickle=True)
        self.res = d["residual_m"].astype(np.float32)            # (F,M,3) metres
        self.mask = d["mask"].astype(bool)                       # (F,M)
        self.pose6d = d["pose6d"].astype(np.float64)             # (F,J,6)
        self.root = d["root_position"].astype(np.float64)        # (F,3)
        self.j_rest = d["j_rest"].astype(np.float64)             # (J,3)
        self.parents = d["parents"].astype(int)                  # (J,)
        self.gR = d["global_R"].astype(np.float64)
        self.gt = d["global_t"].astype(np.float64)
        self.p_bind = c["p_bind"].astype(np.float64)             # (M,3) BVH frame
        self.p_canon = c["p_canon"].astype(np.float64)           # (M,3) Z-up canonical
        self.W = c["W"].astype(np.float64)
        self.F, self.M = self.res.shape[0], self.res.shape[1]
        self.res_norm = np.linalg.norm(self.res, axis=2)
        self.ROUT = rot_x(-DEGREE_BVH_X)                         # BVH -> Z-up
        self.RIN = rot_x(DEGREE_BVH_X)                           # Z-up -> BVH

    def lbs_world(self, f):
        rot = rotmat_from_6d(self.pose6d[f])                     # (J,3,3)
        d = lbs_frame(self.p_bind, self.W, rot, self.j_rest, self.parents, self.root[f])
        d = (self.ROUT @ d.T).T                                  # Z-up
        return (self.gR @ d.T).T + self.gt                       # global alignment

    def _blended_A(self, f):
        rot = rotmat_from_6d(self.pose6d[f]); J = len(self.parents)
        Grest = np.tile(np.eye(4), (J, 1, 1))
        for i in range(J):
            Grest[i, :3, 3] = self.j_rest[i]
        Gp = np.zeros((J, 4, 4))
        for i in range(J):
            L = np.eye(4); L[:3, :3] = rot[i]
            if self.parents[i] < 0:
                L[:3, 3] = self.root[f]; Gp[i] = L
            else:
                L[:3, 3] = self.j_rest[i] - self.j_rest[self.parents[i]]
                Gp[i] = Gp[self.parents[i]] @ L
        skin = Gp @ np.linalg.inv(Grest)
        return np.einsum('vj,jmn->vmn', self.W, skin)[:, :3, :3]

    def joints_world(self, f):
        """world-space joint positions (J,3) for frame f, in the SAME canonical Z-up world
        space as lbs_world (forward kinematics + ROUT + global alignment), so the skeleton
        lines up with the reconstructed markers."""
        rot = rotmat_from_6d(self.pose6d[f]); J = len(self.parents)
        Gp = np.zeros((J, 4, 4))
        for i in range(J):
            L = np.eye(4); L[:3, :3] = rot[i]
            if self.parents[i] < 0:
                L[:3, 3] = self.root[f]; Gp[i] = L
            else:
                L[:3, 3] = self.j_rest[i] - self.j_rest[self.parents[i]]
                Gp[i] = Gp[self.parents[i]] @ L
        pos = (self.ROUT @ Gp[:, :3, 3].T).T                     # Z-up
        return (self.gR @ pos.T).T + self.gt                     # global alignment

    def unpose(self, f, res_world):
        """world residual (M,3) -> canonical-frame displacement (M,3)."""
        A = self._blended_A(f)
        r = res_world @ self.gR                                  # undo global rotation
        r = r @ self.RIN.T                                       # Z-up -> BVH
        d = np.linalg.solve(A, r[..., None])[..., 0]             # A^-1 (un-pose)
        return d @ self.ROUT.T                                   # BVH -> Z-up canonical


def scan(dataset):
    subs = sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(f"{dataset}/*/canonical.npz"))
    shots = {S: sorted(os.path.basename(p)[:-4]
                       for p in glob.glob(f"{dataset}/{S}/shot_*.npz")) for S in subs}
    return subs, shots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DATASET_DEFAULT)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    subs, shots = scan(args.dataset)
    if not subs:
        raise SystemExit(f"no subjects with canonical.npz under {args.dataset}")
    print(f"[viz] dataset {args.dataset}: subjects {subs}")

    import viser
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    g = server.gui

    d_sub = g.add_dropdown("Subject", options=tuple(subs), initial_value=subs[0])
    d_shot = g.add_dropdown("Shot", options=tuple(shots[subs[0]]), initial_value=shots[subs[0]][0])
    d_space = g.add_dropdown("Space", options=(WORLD, CANON), initial_value=WORLD)
    d_global = g.add_dropdown("Remove global", options=(G_OFF, G_RIGID, G_SCALE), initial_value=G_OFF)
    s_frame = g.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
    c_play = g.add_checkbox("Play", initial_value=False)
    s_speed = g.add_slider("Speed (fps)", min=1, max=60, step=1, initial_value=25)
    s_amp = g.add_slider("Display gain", min=0.0, max=10.0, step=0.1, initial_value=0.1)
    s_size = g.add_slider("Point size", min=0.002, max=0.03, step=0.001, initial_value=0.008)
    s_vmax = g.add_slider("Colour scale max (cm)", min=1.0, max=30.0, step=1.0, initial_value=15.0)
    c_obs = g.add_checkbox("Only observed", initial_value=True)
    c_base = g.add_checkbox("Show base/LBS cloud", initial_value=True)
    c_lines = g.add_checkbox("Show displacement lines", initial_value=False)
    txt = g.add_text("Info", initial_value="", disabled=True)

    state = {"shot": None}

    def load(subject, shot):
        state["shot"] = Shot(args.dataset, subject, shot)
        s_frame.max = state["shot"].F - 1
        if s_frame.value > s_frame.max:
            s_frame.value = 0
        txt.value = f"{subject}/{shot}: {state['shot'].F} frames, {state['shot'].M} markers"

    def render(_=None):
        sh = state["shot"]
        if sh is None:
            return
        f = int(s_frame.value)
        sel = sh.mask[f] if c_obs.value else np.ones(sh.M, bool)
        lbs = sh.lbs_world(f)
        obs_f = lbs + sh.res[f]

        if d_global.value != G_OFF:
            fit = sh.mask[f]
            if fit.sum() >= 3:
                s, R, t = kabsch(lbs[fit], obs_f[fit], d_global.value == G_SCALE)
                lbs_view = (s * (R @ lbs.T).T) + t
            else:
                lbs_view = lbs
            r_world = obs_f - lbs_view
        else:
            lbs_view, r_world = lbs, sh.res[f]
        rnorm = np.linalg.norm(r_world, axis=1)

        if d_space.value == WORLD:
            base = lbs_view
            moving = lbs_view + s_amp.value * r_world
        else:
            base = sh.p_canon
            moving = sh.p_canon + s_amp.value * sh.unpose(f, r_world)
        base, moving = base[sel], moving[sel]
        colors = turbo_colors(rnorm[sel] * 100.0, s_vmax.value)

        server.scene.add_point_cloud("/moving", points=moving, colors=colors,
                                     point_size=s_size.value, point_shape="circle")
        if c_base.value:
            server.scene.add_point_cloud("/base", points=base,
                                         colors=np.full((len(base), 3), 130, np.uint8),
                                         point_size=s_size.value * 0.7, point_shape="circle")
        else:
            server.scene.add_point_cloud("/base", points=np.zeros((1, 3), np.float32),
                                         colors=np.zeros((1, 3), np.uint8), point_size=1e-4)
        if c_lines.value and len(base):
            seg = np.stack([base, moving], axis=1)
            server.scene.add_line_segments("/disp", points=seg,
                                           colors=colors[:, None, :].repeat(2, 1), line_width=2.0)
        else:
            server.scene.add_line_segments("/disp", points=np.zeros((1, 2, 3), np.float32),
                                           colors=np.zeros((1, 2, 3), np.uint8), line_width=1e-3)

    @d_sub.on_update
    def _(_):
        d_shot.options = tuple(shots[d_sub.value])
        d_shot.value = shots[d_sub.value][0]      # triggers shot update -> load + render

    @d_shot.on_update
    def _(_):
        load(d_sub.value, d_shot.value); render()

    for w in (d_space, d_global, s_frame, s_amp, s_size, s_vmax, c_obs, c_base, c_lines):
        w.on_update(render)

    load(subs[0], shots[subs[0]][0])

    if args.smoke:
        for sp in (WORLD, CANON):
            d_space.value = sp; render()
        print("[viz] smoke build OK"); return

    render()
    print(f"[viz] open http://localhost:{args.port}  (Ctrl+C to quit)")
    while True:
        if c_play.value and state["shot"] is not None:
            s_frame.value = (int(s_frame.value) + 1) % state["shot"].F
            render()
            time.sleep(1.0 / s_speed.value)
        else:
            time.sleep(0.05)


if __name__ == "__main__":
    main()
