"""Fixed finger poses for the skeleton mesh: the capture has no finger data, so the hand bones stay in the spread rest
pose. curl_hand bends each finger towards the palm, rigidly per bone piece, in the rest frame (numpy only)."""
import numpy as np

POSES = {"grip": ((80, 95, 60), (35, 45)),                  # (MCP, PIP, DIP) of the fingers, (MCP, IP) of the thumb, deg
         "fist": ((95, 105, 70), (45, 60)),
         "relaxed": ((20, 30, 20), (10, 15))}


def _components(faces, keep):
    """Connected components (lists of vertex ids) of the faces whose vertices are all in `keep` (union-find)."""
    parent = {}

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v
    for f in faces:
        if keep[f[0]]:
            for v in f:
                parent.setdefault(int(v), int(v))
            for v in f[1:]:
                a, b = find(int(f[0])), find(int(v))
                if a != b:
                    parent[a] = b
    groups = {}
    for v in parent:
        groups.setdefault(find(v), []).append(v)
    return [np.array(sorted(g)) for g in groups.values()]


def _rotate(x, pivot, k, ang):
    """Rotate points x (N,3) about the axis k through pivot by ang rad (Rodrigues)."""
    k = k / np.linalg.norm(k)
    y = x - pivot
    c, s = np.cos(ang), np.sin(ang)
    return pivot + y * c + np.cross(k, y) * s + np.outer(y @ k, k) * (1 - c)


def curl_hand(V, S, side, mode, dry=False):
    """Bend the fingers of the `side` ("Left"/"Right") hand in V (bone_rest copy, modified in place) into `mode`.
    S: the cache's static.npz. Returns the hand frame (along, palm, lateral) and the finger chains (component ids)."""
    names = [str(n) for n in S["joint_names"]]
    j = names.index(side + "Hand")
    wrist, elbow = S["j_rest"][j], S["j_rest"][names.index(side + "ForeArm")]
    along = (wrist - elbow) / np.linalg.norm(wrist - elbow)
    faces = np.split(S["bone_faces"], np.cumsum(S["bone_face_sizes"])[:-1])
    comps = _components(faces, S["bone_joint"] == j)
    ends = []
    for c in comps:                                                       # endpoints along each piece's long axis
        x = V[c]
        u = np.linalg.svd(x - x.mean(0), full_matrices=False)[2][0]
        t = (x - x.mean(0)) @ u
        e = np.stack([x[t.argmin()], x[t.argmax()]])
        ends.append(e if np.linalg.norm(e[0] - wrist) < np.linalg.norm(e[1] - wrist) else e[::-1])   # (proximal, distal)
    ends = np.array(ends)
    centre = np.array([np.linalg.norm(V[c].mean(0) - wrist) for c in comps])
    carpal = np.zeros(len(comps), bool)
    carpal[np.argsort(centre)[:8]] = True                                 # the 8 carpal bones: nearest the wrist
    parent = np.full(len(comps), -1)
    for i in np.where(~carpal)[0]:                                        # the piece whose distal end meets my proximal end
        d = np.linalg.norm(ends[:, 1] - ends[i, 0], axis=1)
        d[carpal | (np.arange(len(comps)) == i)] = np.inf
        d[np.linalg.norm(ends[:, 1] - wrist, axis=1) >= np.linalg.norm(ends[i, 1] - wrist)] = np.inf
        if d.min() < 0.02:
            parent[i] = d.argmin()
    chains = []
    for root in np.where(~carpal & (parent < 0))[0]:                      # metacarpals
        chain = [root]
        while True:
            kids = np.where(parent == chain[-1])[0]
            if len(kids) == 0:
                break
            chain.append(kids[0])
        chains.append(chain)
    # palm side: the rest pose already curls the fingertips slightly towards the palm
    P = np.concatenate([V[c] for c in comps]) - wrist
    n = np.linalg.svd(P - np.outer(P @ along, along), full_matrices=False)[2][1]
    bend = [(ends[ch[-1], 1] - ends[ch[0], 1]) @ n for ch in chains if len(ch) == 4]
    palm = n * (np.sign(np.mean(bend)) if bend else 1.0)
    info = {"frame": (along, palm, np.cross(along, palm)), "chains": [[int(c) for c in ch] for ch in chains]}
    if dry:
        return info
    fingers, thumb = POSES[mode]
    for ch in chains:
        angles = thumb if len(ch) == 3 else fingers if len(ch) == 4 else ()
        e = ends.copy()
        if len(ch) < 2:
            continue
        k = np.cross(e[ch[1], 1] - e[ch[1], 0], palm)                     # one bend axis per finger, from the straight finger
        for i, ang in zip(range(1, len(ch)), angles):                    # proximal to distal, each bend carries the rest
            pivot = e[ch[i], 0]
            for c in ch[i:]:
                V[comps[c]] = _rotate(V[comps[c]], pivot, k, np.radians(ang))
                e[c] = _rotate(e[c], pivot, k, np.radians(ang))
    return info
