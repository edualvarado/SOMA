import cv2


def compute_matched_indexes(num_frames_per_video, alignment_indexes):
    """Compute the indexes of the frames we want to keep for each video based on the alignment points.

    Args:
        num_frames_per_video: A list of integers representing the number of frames for each video.
        alignment_indexes: A tuple with two lists, where each inner list contains an alignment point for each video.
    """
    per_video_ranges = find_per_video_ranges(alignment_indexes)
    video_indexes = generate_video_indexes(num_frames_per_video)
    cutted_video_indexes = cut_video_indexes(video_indexes, per_video_ranges)

    min_video_length = min([len(indexes) for indexes in cutted_video_indexes])

    subsampled_video_indexes = subsample_video_indexes(cutted_video_indexes, min_video_length)
    return subsampled_video_indexes


def find_per_video_ranges(alignment_indexes):
    """Find the minimum and maximum index for each video based on the alignment points.

    Args:
        alignment_indexes: A list of lists, where each inner list contains an alignment point for each video.

    Returns:
        A list of tuples, where each tuple contains the minimum and maximum index for a video.
    """
    # Find the minimum and maximum index for each video
    per_video_ranges = []
    for i in range(len(alignment_indexes[0])):
        video_alignment = [a[i] for a in alignment_indexes]
        rnge = (min(video_alignment), max(video_alignment))
        per_video_ranges.append(rnge)

    return per_video_ranges


def subsample_video_indexes(video_indexes, desired_frames):
    """Subsample the video indexes to keep a specific number of frames equally distributed, including the first and last frames.

    Args:
        video_indexes: A list of lists, where each inner list contains the indexes of the frames for a video, from 0 to N-1.
        desired_frames: A list of integers representing the number of frames to keep for each video.
    """

    subsampled_video_indexes = []
    for indexes in video_indexes:
        total_frames = len(indexes)

        # Calculate the step size based on the desired number of frames
        step_size = (total_frames - 1) / (desired_frames - 1)

        # Generate the subsampled indexes
        subsampled_indexes = []
        for i in range(desired_frames):
            step = int(round(i * step_size))
            subsampled_indexes.append(indexes[step])

        subsampled_video_indexes.append(subsampled_indexes)

    return subsampled_video_indexes


def cut_video_indexes(video_indexes, per_video_ranges):
    """Cut the video indexes based on the per_video_ranges, cutting the less amount of frames possible.

    Args:
        video_indexes: A list of lists, where each inner list contains the indexes of the frames for a video, from 0 to N-1.
        per_video_ranges: A list of tuples, where each tuple contains the minimum and maximum index for a video.

    Returns:
        A list of lists, where the first and last frame indexes are aligned according to the per_video_ranges.
    """
    # Cut the video indexes based on the per_video_ranges
    cut_video_indexes = []
    for indexes, rnge in zip(video_indexes, per_video_ranges):
        cut_indexes = indexes[rnge[0] : rnge[1]]
        cut_video_indexes.append(cut_indexes)

    return cut_video_indexes


def generate_video_indexes(num_frames_per_video):
    return [list(range(num_frames)) for num_frames in num_frames_per_video]


def find_slowest_fps(video_paths):
    slowest_fps = float("inf")
    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        slowest_fps = min(slowest_fps, fps)
        cap.release()

    return slowest_fps
