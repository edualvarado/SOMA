class Board:
    def detect(self, frame, draw=False):
        raise NotImplementedError

    def detect_all(self, frame, frame_idx, video_path, calibrations, draw=False):
        raise NotImplementedError

    def detect_with_obj_pts(self, frame, draw=False):
        raise NotImplementedError

    def estimate_pose(self, frame, corners, ids, camera_matrix, dist_coeffs):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def from_json(self, path):
        raise NotImplementedError

    def visualize(self, frame, draw=False):
        raise NotImplementedError
