"""Binding of the individual muscle meshes (<S>/layers/tpose/muscle_meshes_tpose/) to the SOMA muscle layer.

Every muscle vertex follows a smooth Gaussian blend of its k nearest muscle-layer vertices (rest T-pose), so the
learned muscle-layer displacement is transferred to all muscle vertices, deep ones included. Computed from the
dataset meshes in well under a second; it replaces the dataset's individual_muscle_to_skin_binding.json, whose
triangle indices do not match the released meshes.
"""
import numpy as np
from scipy.spatial import cKDTree


def bind_to_layer(points, layer_vertices, k=16):
    """(idx, w), both (N, k): the k nearest layer vertices of each point and their normalised Gaussian weights
    (sigma = half the distance to the k-th neighbour, so the blend fades out smoothly at the edge of the set)."""
    d, idx = cKDTree(layer_vertices).query(points, k=k)
    w = np.exp(-(d / (d[:, -1:] / 2.0 + 1e-9)) ** 2)
    return idx, w / w.sum(1, keepdims=True)


def transfer(points, idx, w, layer_disp):
    """points + blended layer displacement. layer_disp: (V, 3) displacement of the layer from its rest pose."""
    return points + (w[..., None] * layer_disp[idx]).sum(1)
