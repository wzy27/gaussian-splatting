import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from utils.general_utils import build_rotation
from scene import GaussianModel


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
        self.direction = direction / torch.norm(direction)
        self.degree = degree
        super().__init__()

    def position(self, t, coord):
        delta = coord - self.center
        current_rotation = R.from_rotvec(
            self.degree * t * self.direction.cpu(), degrees=True
        )
        current_delta = current_rotation.apply(delta.cpu())
        return (
            self.center + torch.from_numpy(current_delta).cuda()
        ).float(), current_rotation


def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    # denom = points_out[..., 3:] + 0.0000001
    # return (points_out[..., :3] / denom).squeeze(dim=0)

    return points_out[..., :3].squeeze(dim=0)


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

    transf_matrix = (
        torch.tensor(
            [
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1],
                [-2, 4, 1],
            ]
        )
        .float()
        .cuda()
    )
    coord = np.array(
        [
            [1, 4, 2],
            [3, 2, 4],
            [0, 2, 6],
        ]
    )
    # print(geom_transform_points(torch.from_numpy(coord), transf_matrix))

    # xyzw
    base_quart = torch.tensor(
        [
            [0.0, 0.0, 0.70710678, 0.70710678],
            # [0.0, 0.38268343, 0.0, 0.92387953],
            # [0.39190384, 0.36042341, 0.43967974, 0.72331741],
        ]
    )
    print("xyzw:", base_quart)
    gaussian_quart = torch.concat((base_quart[:, 3:], base_quart[:, :3]), dim=1)
    print("wxyz:", gaussian_quart)

    base_rotation = build_rotation(gaussian_quart)
    # new_rotation = base_rotation @ transf_matrix[:3, :]

    rot = R.from_matrix(base_rotation.cpu())
    new_rotation = rot.as_quat()  # xyzw quaternion format
    print("xyzw:", new_rotation)
    gaussian_new_rotation = np.concatenate(
        (new_rotation[:, 3:], new_rotation[:, :3]), axis=1
    )  # wxyz quaternion format
    print("wxyz:", gaussian_new_rotation)

    # print("base coord: {}".format(coord))
    # print(
    #     "translation by {}: {}".format(
    #         translation_vector, translation.position(1, coord)
    #     )
    # )
    # print(
    #     "stretch from {} by ratio {:.2f}: {}".format(
    #         s_center, scale, stretch.position(1, coord)
    #     )
    # )
    # print(
    #     "rotation from {} around {} by {} degrees: {}".format(
    #         r_center, r_direction, r_degree, rotation.position(1, coord)
    #     )
    # )

    gaussian = GaussianModel(0)
    gaussian.create_from_pcd()
