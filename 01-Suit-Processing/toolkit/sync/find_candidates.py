from functools import lru_cache

import cv2
import numpy as np
from tqdm import tqdm


@lru_cache(maxsize=16)
def find_candidates(video_path, top_k=5, closeby_threshold=0):
    """Find the top k candidates for a video based on the lightness difference between frames.

    Args:
        video_path: A string representing the path to the video file.
        top_k: An integer representing the number of candidates to find. Defaults to 5.
        closeby_threshold: An integer representing how close two candidates can be. Defaults to 0.

    Returns:
        A list of integers representing the indexes of the top k candidates.
    """
    cap = cv2.VideoCapture(str(video_path))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Finding candidates")

    prev_frame = None
    diffs = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to hsl and calculate the lightness difference
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            gray_diff = (np.mean(gray) - np.mean(prev_frame)) / np.mean(prev_frame)
            diffs.append(gray_diff)

        prev_frame = gray
        pbar.update(1)

    cap.release()
    pbar.close()

    # Find the index of the top k candidates with the lowest difference
    candidates = np.array(diffs)
    candidates = np.argsort(candidates)

    if closeby_threshold:
        candidates = drop_closeby(candidates, closeby_threshold)

    # Keep only the top k candidates
    candidates = candidates[:top_k]

    # Sort them from lowest to highest
    candidates.sort()

    return candidates


def drop_closeby(candidates, threshold=10):
    # Remove candidates that are too close to each other
    new_candidates = [candidates[0]]
    for candidate in candidates[1:]:
        if all(abs(candidate - c) > threshold for c in new_candidates):
            new_candidates.append(candidate)

    return new_candidates
