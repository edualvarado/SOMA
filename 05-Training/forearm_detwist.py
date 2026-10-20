"""Forearm de-twist (2026-10-12), shared by 01/02/03.

The raw Captury BVHs (S2-S5) put a large, noisy roll on the ForeArm joint (twist about the bone axis, mean 70-100 deg,
sd 40-65 deg) that the observed forearm markers mostly do not follow, while LBS applies it rigidly to the whole forearm.
detwist() removes that twist. It is not moved onto the Hand: the hand markers do not follow it either (S3: hand twist vs
ForeArm twist slope -0.02/-0.05, i.e. the observed hand ignores it), and moving it there (2026-10-12 DT_detwist_e20) twisted
the wrist into a candy wrapper.

The residuals were measured against the original-rig marker LBS with the original pose, so a de-twisted run must
convert its targets with old-rig LBS(original pose) - new-rig LBS(de-twisted pose).
"""
import torch


def twist_angle(R, axis):
    """Twist (rad) of rotations R (..., 3, 3) about a unit axis (3,), in the swing-twist split R = R_swing @ R_twist."""
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    return 2 * torch.atan2(v @ axis, 1 + tr)  # quaternion (w, xyz) scaled by 4w: xyz.axis -> v.axis, w -> 1 + trace


def axis_rotation(axis, angle):
    """Rotation matrices (..., 3, 3) about a unit axis (3,) by angle (...) rad (Rodrigues)."""
    K = torch.zeros(3, 3, dtype=axis.dtype, device=axis.device)
    K[0, 1], K[0, 2], K[1, 2] = -axis[2], axis[1], -axis[0]
    K = K - K.T
    s, c = torch.sin(angle)[..., None, None], torch.cos(angle)[..., None, None]
    return torch.eye(3, dtype=axis.dtype, device=axis.device) + s * K + (1 - c) * (K @ K)


def detwist(rot_mats, j_rest, joint_names, parents):
    """Local rotations (B, J, 3, 3) with each ForeArm's twist about its rest bone axis (ForeArm -> Hand in j_rest)
    removed: ForeArm keeps only its swing, the Hand's local rotation is unchanged (rides the de-twisted forearm)."""
    out = rot_mats.clone()
    for side in ("Left", "Right"):
        f, h = joint_names.index(f"{side}ForeArm"), joint_names.index(f"{side}Hand")
        if int(parents[h]) != f:
            raise ValueError(f"{side}Hand is not a child of {side}ForeArm")
        axis = (j_rest[h] - j_rest[f]).to(rot_mats.dtype)
        axis = axis / axis.norm()
        a = twist_angle(rot_mats[:, f], axis)
        out[:, f] = rot_mats[:, f] @ axis_rotation(axis, -a)
    return out


def rotmat_to_sixd(rot_mats):
    """Inverse of sixd_to_rotmat: (..., 3, 3) -> (..., 6), the first two columns."""
    return torch.cat([rot_mats[..., :, 0], rot_mats[..., :, 1]], dim=-1)
