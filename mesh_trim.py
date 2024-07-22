import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
import argparse
import sys
import os
import subprocess, sys
from pathlib import Path
import json


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        assert len(scene_or_mesh.geometry) > 0
        mesh = trimesh.util.concatenate(
            tuple(
                trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                for g in scene_or_mesh.geometry.values()
            )
        )
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh


def get_mesh_info(mesh):
    mesh_max = mesh.vertices.max(axis=0)
    mesh_min = mesh.vertices.min(axis=0)
    mesh_center = (mesh_min + mesh_max) / 2
    mesh_scale = mesh_max - mesh_min
    # mesh_ratio = mesh_scale[2] / mesh_scale[0]

    return mesh_max, mesh_scale


def normalize_and_save(mesh, output_path):
    mesh_max = mesh.vertices.max(axis=0)
    mesh_min = mesh.vertices.min(axis=0)
    mesh_center = (mesh_min + mesh_max) / 2
    mesh_scale = mesh_max - mesh_min

    xyz_normed = (mesh.vertices - mesh_center) / mesh_scale.max()

    direction = np.array([1, 0, 0])
    rotation = R.from_rotvec(270 * direction, degrees=True)
    xyz_rotated = rotation.apply(xyz_normed)

    mesh.vertices = xyz_rotated
    mesh.export(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mesh Trimming")
    parser.add_argument("--data_name", "-d", required=True, type=str)
    parser.add_argument("--threshold", "-t", required=False, type=float, default=1.03)
    args = parser.parse_args(sys.argv[1:])

    ours_path = "/data/nglm005/zhengyu.wen/final/gaussian-splatting/data/Omnizl/{}/meshes/00100000.ply".format(
        args.data_name
    )
    gt_path = "data/CD_test/{}/Scan-matched.ply".format(args.data_name)
    trim_path = "data/CD_test/{0}/{0}-ours-trim.ply".format(args.data_name)

    ours_mesh = as_mesh(trimesh.load(ours_path))
    gt_mesh = as_mesh(trimesh.load(gt_path))

    ours_max, ours_scale = get_mesh_info(ours_mesh)
    _, gt_scale = get_mesh_info(gt_mesh)

    # print("ours:", ours_scale)
    # print("gt:", gt_scale)

    ours_ratio = ours_scale[2] / ours_scale[0]
    gt_ratio = gt_scale[1] / gt_scale[0]

    relative_ratio = ours_ratio / gt_ratio
    ours_threshold_scale = (
        min(relative_ratio, args.threshold) * gt_ratio * ours_scale[0]
    )
    ours_min_z = ours_max[2] - ours_threshold_scale

    # print(
    #     "max: {}, min: {}, thre: {}".format(
    #         ours_max[2], ours_max[2] - ours_scale[2], ours_min_z
    #     )
    # )

    # remove all vertices/faces with z < thre
    keep_mask = ours_mesh.vertices[:, 2] > ours_min_z
    print("vertices total: {}, kept: {}".format(len(keep_mask), keep_mask.sum()))

    face_mask = np.ones(len(ours_mesh.faces), dtype=bool)
    for idx in range(len(ours_mesh.faces)):
        face = ours_mesh.faces[idx]
        for vertex in face:
            if not keep_mask[vertex]:
                face_mask[idx] = False
    print("faces total: {}, kept: {}".format(len(face_mask), face_mask.sum()))

    ours_mesh.update_vertices(keep_mask)
    ours_mesh.update_faces(face_mask)

    normalize_and_save(ours_mesh, trim_path)
