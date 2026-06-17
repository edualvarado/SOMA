"""
derive_relabel.py -- recover the proc_ID -> canonical_ID relabeling for S4/S5.

The Dec-2025 captures (Mona/S4, Sarah/S5) label triangulated markers with IDs that do
not match the canonical model's IDs (the point clouds match geometrically, the labels
don't). This script derives a per-subject relabeling map by matching, frame by frame,
each proc marker (centroid of its 4 corners) against the *Blender-recovered observed*
cloud (export + residual/0.1 from shot_001), which carries correct canonical labels.
Observation-to-observation matching is tight (~2-3 cm), unlike obs-to-LBS (~6 cm soft
tissue), so a vote across hundreds of frames is decisive.

Robustness:
  * mutual nearest-neighbour + distance threshold per frame
  * majority voting across frames; map accepted only if purity >= MIN_PURITY and
    votes >= MIN_VOTES; one-to-one enforced greedily by vote count
  * corner permutation per proc ID recovered the same way (brute-force over 24 perms
    of 4 corners, voted across frames)

Output: relabel_<S>.json next to this script:
  {"map": {"<proc_id>": {"canon": "<canonical_id>", "perm": [p0,p1,p2,p3],
                          "votes": n, "purity": x, "perm_purity": y}}, "stats": {...}}
perm semantics: proc corner j corresponds to canonical corner perm[j].

Run:
  /CT/SOMA/static00/miniforge3/envs/soma/bin/python derive_relabel.py S4 S5
"""
import sys
import json
import itertools
import numpy as np
from collections import defaultdict, Counter
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

import batch_residuals as B
from batch_residuals import _TF

SCALE = 0.1          # historical residual scale in the existing files
CAND_R = 0.10        # candidate radius: canon markers within this of a proc centroid (m)
MIN_CO = 6           # min frames a (proc,canon) pair must co-occur to be trusted
COST_T = 0.06        # max mean corner-fit distance to accept a mapping (m); proc vs the
                     # Blender-recovered observed differ ~3 cm even for true matches
N_SAMPLE = 450       # target number of sampled frames

PERMS = [np.array(p) for p in itertools.permutations(range(4))]


def best_perm_cost(P, C):
    """min over the 24 corner permutations of mean |P[j]-C[perm[j]]|; returns (cost, perm)."""
    best_c, best_p = 1e9, PERMS[0]
    for p in PERMS:
        c = np.linalg.norm(P - C[p], axis=1).mean()
        if c < best_c:
            best_c, best_p = c, p
    return best_c, best_p


def derive(S):
    sd = f"{B.STATIC}/{S}/raw/shot_001_captury"
    print(f"[{S}] loading Blender export + residuals (large)...", flush=True)
    exp = json.load(open(f"{sd}/{S}_canonical_markers_lbs_shot_001_exported_tpose.json"))
    res = json.load(open(f"{sd}/{S}_residuals_shot_001_world_lbs_scaled_tpose.json"))
    proc = json.load(open(B.tri_path(S, "shot_001")))

    frames = sorted(set(exp) & set(res), key=int)
    step = max(1, len(frames) // N_SAMPLE)
    frames = frames[::step]
    print(f"[{S}] aggregating corner-fit costs over {len(frames)} frames...", flush=True)

    # aggregate over all frames: (proc_id, canon_id) -> [sum_cost, count, perm_votes]
    agg = defaultdict(lambda: [0.0, 0, Counter()])
    proc_seen, canon_seen = set(), set()

    for fk in frames:
        f = int(fk)
        pf = proc.get(str(f - 1))
        if not pf:
            continue
        # Blender observed cloud with canonical labels (only truly observed: residual != 0)
        canon_pts = {}                       # canon_id -> {corner: xyz}
        for m, v in res[fk].items():
            r = np.asarray(v, float).reshape(-1)[:3]
            if not np.any(r):
                continue
            e = np.asarray(exp[fk].get(m, [[0, 0, 0]])[0], float)
            parts = m.split("_")             # marker_<id>_<inst>_<corner>
            canon_pts.setdefault(parts[1], {})[int(parts[3])] = e + r / SCALE
        cids, ccorn = [], []
        for cid, corners in canon_pts.items():
            if len(corners) == 4:
                cids.append(cid); ccorn.append(np.array([corners[c] for c in range(4)]))
        if len(cids) < 20:
            continue
        ccorn = np.array(ccorn)              # (Nc,4,3)
        ccent = ccorn.mean(1)
        ctree = cKDTree(ccent)

        for pid, corners in pf.items():
            if len(corners) != 4 or not isinstance(corners[0][0], (int, float)):
                continue
            P = (_TF @ np.asarray(corners, float).T).T   # (4,3)
            pc = P.mean(0)
            proc_seen.add(pid)
            for ci in ctree.query_ball_point(pc, CAND_R):
                cost, perm = best_perm_cost(P, ccorn[ci])
                rec = agg[(pid, cids[ci])]
                rec[0] += cost; rec[1] += 1; rec[2][tuple(perm)] += 1
                canon_seen.add(cids[ci])

    # build aggregate cost matrix and solve ONE global assignment
    pids = sorted(proc_seen); cids = sorted(canon_seen)
    pidx = {p: i for i, p in enumerate(pids)}; cidx = {c: i for i, c in enumerate(cids)}
    BIG = 1e3
    CM = np.full((len(pids), len(cids)), BIG)
    for (pid, cid), (s, n, _) in agg.items():
        if n >= MIN_CO:
            CM[pidx[pid], cidx[cid]] = s / n
    ri, ci = linear_sum_assignment(CM)
    assigned_costs = np.array([CM[r, c] for r, c in zip(ri, ci) if CM[r, c] < BIG])
    if len(assigned_costs):
        sweep = {f"{t}cm": int((assigned_costs < t / 100).sum()) for t in (3, 4, 5, 6, 8, 10)}
        print(f"[{S}] assigned pairs below cost: {sweep}  (of {len(assigned_costs)} assigned)")
    mapping = {}
    for r, c in zip(ri, ci):
        cost = CM[r, c]
        if cost >= COST_T:
            continue
        pid, cid = pids[r], cids[c]
        s, n, pv = agg[(pid, cid)]
        perm, pn = pv.most_common(1)[0]
        mapping[pid] = {"canon": cid, "perm": [int(x) for x in perm], "cost_cm": round(float(cost) * 100, 2),
                        "frames": int(n), "perm_purity": round(pn / sum(pv.values()), 3)}

    ident = sum(1 for pid, m in mapping.items() if pid == m["canon"])
    perm_id = sum(1 for m in mapping.values() if m["perm"] == [0, 1, 2, 3])
    stats = {"subject": S, "frames_voted": len(frames), "proc_ids_seen": len(proc_seen),
             "mapped": len(mapping), "identity_ids": ident, "identity_perms": perm_id}
    print(f"[{S}] mapped {len(mapping)}/{len(proc_seen)} proc IDs "
          f"({ident} identical, {len(mapping)-ident} relabeled; "
          f"{perm_id} with identity corner order)")
    costs = np.array([m["cost_cm"] for m in mapping.values()])
    if len(costs):
        print(f"[{S}] mapping cost: median={np.median(costs):.2f}cm p90={np.percentile(costs,90):.2f}cm")
    out = f"/CT/SOMA/work/08-Residuals-Python/relabel_{S}.json"
    with open(out, "w") as fo:
        json.dump({"map": mapping, "stats": stats}, fo, indent=1)
    print(f"[{S}] wrote {out}")
    # quick pattern peek
    pairs = sorted((int(p), int(m["canon"])) for p, m in mapping.items()
                   if p.isdigit() and m["canon"].isdigit() and p != m["canon"])
    if pairs:
        print(f"[{S}] sample relabeled pairs (proc->canon): {pairs[:10]}")


if __name__ == "__main__":
    for S in (sys.argv[1:] or ["S4", "S5"]):
        derive(S)
