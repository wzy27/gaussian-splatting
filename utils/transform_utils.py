import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch


class Transform:
    def __init__(self) -> None:
        pass

    def position(self, t, coord):
        print("Undefined")


class Translation(Transform):
    def __init__(self, vec) -> None:
        self.v = vec
        super().__init__()

    def position(self, t, coord):
        return coord + t * self.v


class Stretch(Transform):
    def __init__(self, center, final_ratio) -> None:
        self.center = center
        self.ratio = final_ratio
        super().__init__()

    def position(self, t, coord):
        delta = coord - self.center
        current_ratio = math.exp(t * math.log(self.ratio))
        return self.center + delta * current_ratio


class Rotation(Transform):
    def __init__(self, center, direction, degree) -> None:
        self.center = center
        self.direction = direction / np.linalg.norm(direction)
        self.degree = degree
        super().__init__()

    def position(self, t, coord):
        delta = coord - self.center
        current_rotation = R.from_rotvec(self.degree * t * self.direction, degrees=True)
        current_delta = current_rotation.apply(delta)
        return self.center + current_delta


if __name__ == "__main__":
    coord = np.array([3, 2, 4])

    translation_vector = np.array([1, 2, 3])
    translation = Translation(translation_vector)

    s_center = np.array([2, 1, 2])
    scale = 0.8
    stretch = Stretch(s_center, scale)

    r_center = np.array([2, 2, 2])
    r_direction = np.array([0, 3, 0])
    r_degree = 180
    rotation = Rotation(r_center, r_direction, r_degree)

    print("base coord: {}".format(coord))
    print(
        "translation by {}: {}".format(
            translation_vector, translation.position(1, coord)
        )
    )
    print(
        "stretch from {} by ratio {:.2f}: {}".format(
            s_center, scale, stretch.position(1, coord)
        )
    )
    print(
        "rotation from {} around {} by {} degrees: {}".format(
            r_center, r_direction, r_degree, rotation.position(1, coord)
        )
    )
