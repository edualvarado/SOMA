"""
derive_relabel_lbs.py -- recover the proc_ID -> canonical_ID relabeling by matching the
proc triangulation against the *LBS marker cloud* (canonical markers posed by the shot BVH),
instead of the Blender-recovered observed cloud used by derive_relabel.py.

Why: for S4 (Mona) the Blender reference was built from the OLD (bad) triangulation, so the
Blender-obs-based map systematically mapped proc IDs to neighbour markers (~11 cm residual).
An oracle chamfer test on the NEW triangulation shows observed corners sit only ~2.5 cm from
the nearest LBS marker -- i.e. the geometry is clean and only the LABELS are wrong. The LBS
cloud carries the TRUE canonical labels, is reproducible from the BVH, and (one-to-one,
corner-fit Hungarian, voted over many frames) recovers the correct correspondence robustly.

Pipeline:
  1. Robust ICP (no labels) to align the proc cloud to the LBS cloud for shot_001.
  2. Per sampled frame: candidate LBS markers within CAND_R of each proc-marker centroid;
     best-of-24 corner-permutation fit cost; aggregate (proc_id, canon_id) -> mean cost + perm votes.
  3. ONE global one-to-one assignment (Hungarian); accept pairs below COST_T.

Output: relabel_<S>.json  (same schema as derive_relabel.py; consumed by batch_residuals.py).

Run:
  /CT/SOMA/static00/miniforge3/envs/soma/bin/python derive_relabel_lbs.py S4
"""
import sys, json, itertools
import numpy as np
from collections import defaultdict, Counter
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

import batch_residuals as B
from batch_residuals import _TF
from recreate_residuals import quat_wxyz_to_matrix, lbs_frame, rot_x, DEGREE_BVH_X
from pymotion.io.bvh import BVH

CAND_R = 0.08        # candidate radius LBS<->proc centroid (m); oracle chamfer ~2.5 cm
MIN_CO = 6           # min co-occurrence frames to trust a (proc,canon) pair
COST_T = 0.045       # max mean corner-fit distance to accept (m); true LBS matches ~2-3 cm
N_SAMPLE = 450
ICP_ITERS = 8

PERMS = [np.array(p) for p in itertools.permutations(range(4))]


def best_perm_cost(P, C):
    best_c, best_p = 1e9, PERMS[0]
    for p in PERMS:
        c = np.linalg.norm(P - C[p], axis=1).mean()
        if c < best_c:
            best_c, best_p = c, p
    return best_c, best_p


def derive(S):
    model = B.load_subject_model(S)
    markers = model["markers"]; mid_map = model["mid_map"]
    p_bind, W = model["p_bind"], model["W"]
    sh = "shot_001"
    bvh = BVH(); bvh.load(B.bvh_path(S, sh))
    lr, lp, parents, offsets, _, _ = bvh.get_data()
    parents = parents.copy(); parents[0] = -1
    Nb, J = lr.shape[0], len(parents)
    bscale = 0.001 if np.abs(offsets).max() > 10.0 else 1.0
    offsets = offsets * bscale
    root_all = lp[:, 0, :] * bscale
    j_rest = np.zeros_like(offsets)
    for i in range(J):
        j_rest[i] = offsets[i] if parents[i] < 0 else j_rest[parents[i]] + offsets[i]
    dom = W.argmax(1)
    t_align = np.median(p_bind - j_rest[dom], axis=0)
    if np.linalg.norm(t_align) > 0.25:
        j_rest = j_rest + t_align

    def lbsw(bf):
        rot = quat_wxyz_to_matrix(lr[bf])
        return (rot_x(-DEGREE_BVH_X) @ lbs_frame(p_bind, W, rot, j_rest, parents, root_all[bf]).T).T

    # LBS targets per (canon_id, instance) that have all 4 corners present
    # targets[k] = (cid, np.array(4,3) of marker indices into the LBS array)
    target_meta = []   # (cid, [idx0,idx1,idx2,idx3])
    for cid, cmap in mid_map.items():
        if all(c in cmap for c in range(4)):
            # one entry per instance present in all corners
            insts = set.intersection(*[{inst for inst, _ in cmap[c]} for c in range(4)])
            for inst in insts:
                idxs = [next(mi for ins, mi in cmap[c] if ins == inst) for c in range(4)]
                target_meta.append((cid, np.array(idxs)))
    print(f"[{S}] {len(target_meta)} LBS targets (canon_id x instance with 4 corners)", flush=True)

    print(f"[{S}] loading proc {sh} (large)...", flush=True)
    proc = json.load(open(B.tri_path(S, sh)))
    Np = max(int(k) for k in proc) + 1
    F = min(Np, Nb - 1)
    frames = list(range(100, F, max(1, (F - 100) // N_SAMPLE)))
    print(f"[{S}] sampling {len(frames)} frames", flush=True)

    # ---- robust ICP (no labels): align proc cloud -> LBS cloud ----
    gR, gt = np.eye(3), np.zeros(3)
    for it in range(ICP_ITERS):
        X, Y = [], []
        for i in frames[::3]:
            lw = lbsw(i + 1)
            tree = cKDTree(lw)
            pf = proc.get(str(i), {})
            for mid, corners in pf.items():
                if len(corners) != 4:
                    continue
                arr = (gR @ (_TF @ np.asarray(corners, float).T)).T + gt
                dd, nn = tree.query(arr)
                for c in range(4):
                    if dd[c] < CAND_R:
                        X.append((_TF @ np.asarray(corners[c], float)))
                        Y.append(lw[nn[c]])
        if len(X) < 100:
            break
        R, t = B.umeyama_rigid(np.asarray(X), np.asarray(Y))
        gR, gt = R, t
        err = np.linalg.norm((R @ np.asarray(X).T).T + t - np.asarray(Y), axis=1).mean()
        print(f"[{S}] ICP it{it}: pairs={len(X)} mean={err*100:.2f}cm", flush=True)

    # ---- aggregate corner-fit costs proc_id x canon_id ----
    agg = defaultdict(lambda: [0.0, 0, Counter()])
    proc_seen = set()
    for i in frames:
        lw = (gR @ lbsw(i + 1).T).T + gt
        cents = np.array([lw[idxs].mean(0) for _, idxs in target_meta])
        ctree = cKDTree(cents)
        pf = proc.get(str(i), {})
        for pid, corners in pf.items():
            if len(corners) != 4 or not isinstance(corners[0][0], (int, float)):
                continue
            P = (_TF @ np.asarray(corners, float).T).T
            proc_seen.add(pid)
            for ti in ctree.query_ball_point(P.mean(0), CAND_R):
                cid, idxs = target_meta[ti]
                cost, perm = best_perm_cost(P, lw[idxs])
                rec = agg[(pid, cid)]
                rec[0] += cost; rec[1] += 1; rec[2][tuple(perm)] += 1

    pids = sorted(proc_seen)
    cids = sorted({cid for (_, cid) in agg})
    pidx = {p: i for i, p in enumerate(pids)}; cidx = {c: i for i, c in enumerate(cids)}
    BIG = 1e3
    CM = np.full((len(pids), len(cids)), BIG)
    for (pid, cid), (s, n, _) in agg.items():
        if n >= MIN_CO:
            CM[pidx[pid], cidx[cid]] = s / n
    ri, ci = linear_sum_assignment(CM)
    assigned = np.array([CM[r, c] for r, c in zip(ri, ci) if CM[r, c] < BIG])
    if len(assigned):
        sweep = {f"{t}cm": int((assigned < t / 100).sum()) for t in (2, 3, 4, 5, 6, 8)}
        print(f"[{S}] assigned pairs below cost: {sweep}  (of {len(assigned)} assigned)")
    mapping = {}
    for r, c in zip(ri, ci):
        cost = CM[r, c]
        if cost >= COST_T:
            continue
        pid, cid = pids[r], cids[c]
        s, n, pv = agg[(pid, cid)]
        perm, pn = pv.most_common(1)[0]
        mapping[pid] = {"canon": cid, "perm": [int(x) for x in perm],
                        "cost_cm": round(float(cost) * 100, 2), "frames": int(n),
                        "perm_purity": round(pn / sum(pv.values()), 3)}
    ident = sum(1 for pid, m in mapping.items() if pid == m["canon"])
    stats = {"subject": S, "method": "lbs", "frames_voted": len(frames),
             "proc_ids_seen": len(proc_seen), "mapped": len(mapping), "identity_ids": ident}
    costs = np.array([m["cost_cm"] for m in mapping.values()])
    print(f"[{S}] mapped {len(mapping)}/{len(proc_seen)} proc IDs ({ident} identical); "
          f"cost median={np.median(costs):.2f}cm p90={np.percentile(costs,90):.2f}cm", flush=True)
    out = f"/CT/SOMA/work/08-Residuals-Python/relabel_{S}.json"
    json.dump({"map": mapping, "stats": stats}, open(out, "w"), indent=1)
    print(f"[{S}] wrote {out}", flush=True)


if __name__ == "__main__":
    for S in (sys.argv[1:] or ["S4"]):
        derive(S)
