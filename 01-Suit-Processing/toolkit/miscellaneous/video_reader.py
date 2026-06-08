from functools import lru_cache

import av
import cv2
import numpy as np


class VideoReader:
    """
    A class to read video frames from a file.

    Args:
        video_path: A string or `Path` representing the path to the video file.
        safe_mode: A boolean indicating whether to use safe mode to read frames.
                    In safe mode, it uses the OpenCV VideoCapture and loops through the frames to get the desired frame.
                        This is very slow since it reads all the frames until the desired frame, but reliable.
                    In unsafe mode, it uses the PyAV library to seek to the desired frame directly.
                        This is faster but may not work for all video formats. In our case, it always worked so far.
    """

    def __init__(self, video_path, safe_mode=False):
        self.video_path = video_path
        if safe_mode:
            self.cap = cv2.VideoCapture(video_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.get_frame = self.get_frame_safe
        else:
            self.cap = av.open(video_path)
            self.total_frames = int(self.cap.streams.video[0].frames)
            if self.total_frames == 0:
                self.total_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0

    def next_frame(self):
        if self.current_frame >= self.total_frames:
            return False, None

        return self.get_frame(self.current_frame + 1)

    def previous_frame(self):
        if self.current_frame <= 0:
            return False, None

        return self.get_frame(self.current_frame - 1)

    @lru_cache(maxsize=16)
    def get_frame_safe(self, frame_number):
        # Clip the frame number to the valid range
        frame_number = max(0, min(frame_number, self.total_frames - 1))

        # If the desired frame is before the current frame, reset the video
        if frame_number < self.current_frame:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0

        # Grab the frames until the desired frame
        for _ in range(self.current_frame, frame_number):
            self.cap.grab()
            self.current_frame += 1

        success, frame = self.cap.read()
        self.current_frame += 1
        return success, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @lru_cache(maxsize=16)
    def get_frame(self, frame_number):
        # Clip the frame number to the valid range
        frame_number = max(0, min(frame_number, self.total_frames - 1))

        framerate = self.cap.streams.video[0].average_rate  # get the frame rate
        time_base = self.cap.streams.video[0].time_base  # get the time base

        sec = int(frame_number / framerate)  # timestamp for that frame_num

        self.cap.seek(sec * 1000000, backward=True)  # seek to that nearest timestamp
        frame = next(self.cap.decode(video=0))  # get the next available frame

        sec_frame = int(frame.pts * time_base * framerate)  # get the proper key frame number of that timestamp

        for _ in range(sec_frame, frame_number):
            frame = next(self.cap.decode(video=0))

        return True, np.array(frame.to_image()) if frame else None
