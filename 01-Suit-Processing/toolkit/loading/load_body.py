import json

from ..common import Pose


def load_body(file):
    with open(file, "r") as f:
        body = json.load(f)

    outdict = {}
    for timestep, data in body.items():
        timedict = {}

        for joint, values in data.items():
            rvec, tvec = values
            pose = Pose(rvec, tvec, parent="world", child=joint)

            timedict[joint] = pose

        outdict[int(timestep)] = timedict

    return outdict


def load_skeleton(file):
    with open(file, "r") as f:
        skeleton = json.load(f)

    return skeleton
