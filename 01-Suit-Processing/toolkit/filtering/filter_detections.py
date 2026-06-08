import numpy as np


def intersect_detections_by_index(left_detections, right_detections):
    corners_left, corners_right, ids = [], [], []

    for (c_left, i_left), (c_right, i_right) in zip(left_detections, right_detections):
        common_ids = set(i_left) & set(i_right)
        cl, cr, id_ = [], [], []

        for i in common_ids:
            pos_left = np.where(i_left == i)[0][0]
            pos_right = np.where(i_right == i)[0][0]

            cl.append(c_left[pos_left])
            cr.append(c_right[pos_right])
            id_.append(i)

        cl = np.array(cl, dtype=np.float32)
        cr = np.array(cr, dtype=np.float32)
        id_ = np.array(id_, dtype=np.int32)

        corners_left.append(cl)
        corners_right.append(cr)
        ids.append(id_)

    return corners_left, corners_right, ids


def filter_single_detection(detections, min_corners=5):
    return {k: v for k, v in detections.items() if len(v["corners"]) > min_corners}


def filter_closeby_frames(detections, threshold=10, use_motion=False):
    keys = list(detections.keys())

    # If `use_motion` is True, sort the keys such that the frames with the least motion are first
    if use_motion is not None:
        # Each key in the detections dictionary has a `meta` key with `displacement` and `overlap`
        def get_cost(idx):
            detection = detections[idx]
            displacement, overlap = detection["meta"]["displacement"], detection["meta"]["overlap"]
            return displacement if overlap > 0.5 else float("inf")

        keys = sorted(keys, key=get_cost)
    else:
        keys = sorted(keys)

    selected_keys = [keys[0]]

    for key in keys[1:]:
        # Add key if it is more than `threshold` frames away from any of the selected keys
        if all(abs(key - k) > threshold for k in selected_keys):
            selected_keys.append(key)

    to_delete = set(keys) - set(selected_keys)
    to_delete = sorted(list(to_delete), reverse=True)

    print(f"Dropping {len(to_delete)} frames out of {len(detections)}")
    for key in to_delete:
        del detections[key]

    return detections


def intersect_detections_by_time(left_detections, right_detections):
    common_frame_nums = set(left_detections.keys()) & set(right_detections.keys())
    print(f"Common frames: {len(common_frame_nums)}")

    left_detections = {k: left_detections[k] for k in common_frame_nums}
    right_detections = {k: right_detections[k] for k in common_frame_nums}

    return left_detections, right_detections, common_frame_nums
