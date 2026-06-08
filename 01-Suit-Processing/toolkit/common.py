from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Intrinsics:
    matrix: np.ndarray
    distortion: np.ndarray

    def to_dict(self):
        return {
            "K": self.matrix.tolist(),
            "d": self.distortion.tolist(),
        }


@dataclass
class Pose:
    """
    Pose class that stores the rotation and translation of an object.
    Given a point in frame `child`, then `self @ point` is the same point in frame `parent`.
    We will ofter use the notation `wTc` to denote a pose with `w` as the parent and `c` as the child.
    """

    rvec: np.ndarray
    tvec: np.ndarray
    parent: str = None
    child: str = None

    def __post_init__(self):
        if isinstance(self.rvec, list):
            self.rvec = np.array(self.rvec)
            self.tvec = np.array(self.tvec)

        self.rvec = self.rvec.reshape(3, 1)
        self.tvec = self.tvec.reshape(3, 1)

    @property
    def matrix(self):
        R, t = cv2.Rodrigues(self.rvec)[0], self.tvec

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        return T

    @property
    def R(self):
        return cv2.Rodrigues(self.rvec)[0]

    def to_list(self):
        return [self.rvec.flatten().tolist(), self.tvec.flatten().tolist()]

    @classmethod
    def from_matrix(cls, T, **kwargs):

        if T.shape == (3, 3):
            T = np.pad(T, ((0, 1), (0, 1)), mode="constant", constant_values=0)
            T[-1, -1] = 1

        R = T[:3, :3]
        t = T[:3, 3]

        rvec, _ = cv2.Rodrigues(R)
        return cls(rvec, t, **kwargs)

    @classmethod
    def from_quaternion(cls, q, t, **kwargs):
        R = Rotation.from_quat(q).as_matrix()
        tvec = np.array(t).reshape(3, 1)
        rvec, _ = cv2.Rodrigues(R)
        return cls(rvec, tvec, **kwargs)

    def inv(self):
        R, t = cv2.Rodrigues(self.rvec)[0], self.tvec

        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = -R.T @ t.flatten()

        new_parent, new_child = self.child, self.parent
        return Pose.from_matrix(T, parent=new_parent, child=new_child)

    def __matmul__(self, other):
        if isinstance(other, np.ndarray):
            return self.matrix @ other

        if self.child != other.parent:
            raise ValueError(f"Cannot multiply poses with different frames: {self.child} and {other.parent}")

        T1 = self.matrix
        T2 = other.matrix
        parent, child = self.parent, other.child
        return Pose.from_matrix(T1 @ T2, parent=parent, child=child)

    def right_handed(self):
        M = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        new_T = M @ self.matrix @ M
        return Pose.from_matrix(new_T, parent=self.parent, child=self.child)
