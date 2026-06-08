from typing import List

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..common import Pose


def weighted_average_transform(transformations: List[Pose], weights=None, outlier_threshold=1.0):
    if weights is None:
        weights = np.ones(len(transformations))

    weights = np.array(weights, dtype=np.float32)
    weights /= weights.sum()

    # Find the outliers
    if outlier_threshold and len(transformations) > 3:
        outliers = find_outliers(transformations, threshold=outlier_threshold)
    else:
        outliers = []

    if len(outliers) > len(transformations) / 3:
        print(f"{len(outliers)/len(transformations)*100:.1f}% of the poses are outliers")

    if len(outliers) == len(transformations):
        print("All the poses are outliers, increasing the threshold")
        outliers = find_outliers(transformations, threshold=3.0)

    # Filter out outliers
    filtered_transforms = [t for i, t in enumerate(transformations) if i not in outliers]
    filtered_weights = [w for i, w in enumerate(weights) if i not in outliers]

    # Normalize the weights
    filtered_weights = np.array(filtered_weights)
    if filtered_weights.sum() > 0.0:
        filtered_weights /= filtered_weights.sum()
    else:
        raise ValueError("All weights are zero")

    # Compute the weighted average of the poses
    avg_t, avg_r, _, _ = average_transform_and_variance(filtered_transforms, filtered_weights)

    # Convert the average back to a transformation matrix
    T = np.eye(4)
    T[:3, :3] = avg_r.as_matrix()
    T[:3, 3] = avg_t

    avg_pose = Pose.from_matrix(T, parent=transformations[0].parent, child=transformations[0].child)

    return avg_pose, outliers


def find_outliers(transformations, threshold=1.5, return_variance=False):
    translations = np.array([t.matrix[:3, 3] for t in transformations])
    rotations = R.from_quat([R.from_matrix(t.matrix[:3, :3]).as_quat() for t in transformations])

    # Step 1: Detect translation outliers
    median_translation = np.median(translations, axis=0)
    translation_distances = np.linalg.norm(translations - median_translation, axis=1)
    translation_iqr = np.subtract(*np.percentile(translation_distances, [75, 25]))
    translation_outlier_threshold = threshold * translation_iqr
    translation_outliers = translation_distances > (np.median(translation_distances) + translation_outlier_threshold)

    # Step 2: Detect rotation outliers using the geodesic distance
    median_rotation = R.mean(R.from_quat([rot.as_quat() for rot in rotations]))  # Approximating the median rotation
    rotation_distances = np.array([geodesic_distance(rot, median_rotation) for rot in rotations])
    rotation_iqr = np.subtract(*np.percentile(rotation_distances, [75, 25]))
    rotation_outlier_threshold = threshold * rotation_iqr
    rotation_outliers = rotation_distances > (np.median(rotation_distances) + rotation_outlier_threshold)

    # Step 3: Combine detection results
    outliers = np.logical_or(translation_outliers, rotation_outliers)

    # Return the indices of the outliers
    outlier_idx = np.where(outliers)[0]

    return outlier_idx


def compute_variance(transformations):
    translations = np.array([t.matrix[:3, 3] for t in transformations])
    rotations = R.from_quat([R.from_matrix(t.matrix[:3, :3]).as_quat() for t in transformations])

    mean_translation = np.mean(translations, axis=0)
    translation_distances = np.linalg.norm(translations - mean_translation, axis=1)

    mean_rotation = R.mean(R.from_quat([rot.as_quat() for rot in rotations]))  # Approximating the mean rotation
    rotation_distances = np.array([geodesic_distance(rot, mean_rotation) for rot in rotations])

    translation_variance = np.mean(np.square(translation_distances))
    rotation_variance = np.mean(np.square(rotation_distances))

    return translation_variance, rotation_variance


def rot_tvec_from_matrix(T):
    r = T[:3, :3]
    t = T[:3, 3]
    r = R.from_matrix(r)
    return r, t


def average_transform_and_variance(transformations, weights):
    r_t = [rot_tvec_from_matrix(t.matrix) for t in transformations]

    r, ts = [i[0] for i in r_t], [i[1] for i in r_t]
    avg_t, avg_r = average_transform(r, ts, weights)

    # Translation variance
    var_t = sum(w * np.square(t - avg_t) for w, t in zip(weights, ts)).mean()

    # Rotation variance
    var_r = sum(w * geodesic_distance(r, avg_r) ** 2 for w, r in zip(weights, r)).mean()
    return avg_t, avg_r, var_t, var_r


def average_transform(rots, transl, weights):
    # Convert into a single Rotation object
    rots = R.from_quat([i.as_quat() for i in rots])

    # Average translation
    avg_t = sum(w * t for w, t in zip(weights, transl))

    # Compute the average quaternion
    avg_r = rots.mean(weights=weights)

    return avg_t, avg_r


def geodesic_distance(r1, r2):
    """Compute the angular distance between two rotations"""
    R1, R2t = r1.as_matrix(), r2.as_matrix().T
    tr = np.trace(R1 @ R2t)
    cos = (tr - 1) / 2
    return np.arccos(np.clip(cos, -1, 1))
