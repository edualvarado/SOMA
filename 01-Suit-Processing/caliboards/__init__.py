import sys
sys.path.append(".")

import json

from .arucoboard import ArucoBoard
from .charucoboard import ChArUcoBoard
from .suit import Suit

class BoardFactory:
    @staticmethod
    def create_board(board_type, **kwargs):
        if board_type == "charuco":
            return ChArUcoBoard(**kwargs)
        elif board_type == "aruco":
            return ArucoBoard(**kwargs)
        elif board_type == "charuco-suit":
            return Suit(**kwargs)
        else:
            raise ValueError(f"Unknown board type {board_type}")

    @staticmethod
    def from_json(path):
        with open(path, "r") as file:
            data = json.load(file)
        return BoardFactory.create_board(**data)
