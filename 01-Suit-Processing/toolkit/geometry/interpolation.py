from datetime import datetime
from typing import Dict, List

import numpy as np
from toolkit.common import Pose

from .average import weighted_average_transform


def find_closest_timestamps(timestamp_list: List[datetime], query_timestamp: datetime) -> tuple[datetime, datetime]:
    """
    Find the two timestamps in the list that are closest to the query timestamp,
    one before and one after.
    """
    before = None
    after = None
    for ts in sorted(timestamp_list):
        if ts <= query_timestamp:
            before = ts
        elif ts > query_timestamp and after is None:
            after = ts
            break
    return before, after


def interpolate_poses(poses: Dict[datetime, Pose], query_timestamp: datetime) -> Pose:
    """
    Returns a continuous function that given a datetime interpolates between the
    closest points in the dictionary using a weighted average.

    Parameters:
    - poses: dict, with datetime objects as keys and Pose objects as values.
    - query_timestamp: datetime, the timestamp for which to interpolate the pose.
    """
    timestamps = list(poses.keys())
    before, after = find_closest_timestamps(timestamps, query_timestamp)

    if before is None:
        raise ValueError("Query timestamp is before all timestamps in the dictionary.")
    if after is None:
        raise ValueError("Query timestamp is after all timestamps in the dictionary.")

    before_pose = poses[before]
    after_pose = poses[after]

    # Calculate weights based on temporal proximity
    total_interval = (after - before).total_seconds()
    weight_after = (query_timestamp - before).total_seconds() / total_interval
    weight_before = 1 - weight_after

    # Perform weighted average
    interpolated_pose, _ = weighted_average_transform([before_pose, after_pose], [weight_before, weight_after])

    return interpolated_pose


def interpolate_vectors(arrays: Dict[datetime, np.ndarray], query_timestamp: datetime) -> np.ndarray:
    """
    Returns a continuous function that given a datetime interpolates between the
    closest points in the dictionary using a weighted average.

    Parameters:
    - array: list, with numpy arrays as elements.
    - query_timestamp: datetime, the timestamp for which to interpolate the pose.
    """
    timestamps = list(arrays.keys())
    before, after = find_closest_timestamps(timestamps, query_timestamp)

    if before is None:
        raise ValueError("Query timestamp is before all timestamps in the dictionary.")
    if after is None:
        raise ValueError("Query timestamp is after all timestamps in the dictionary.")

    before_array = arrays[before]
    after_array = arrays[after]

    # Calculate weights based on temporal proximity
    total_interval = (after - before).total_seconds()
    weight_after = (query_timestamp - before).total_seconds() / total_interval
    weight_before = 1 - weight_after

    # Perform weighted average
    interpolated_array = before_array * weight_before + after_array * weight_after

    return interpolated_array
