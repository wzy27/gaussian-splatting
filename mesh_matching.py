import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
import argparse
import sys


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


def normalize_mesh(input_path, output_path, degree):
    mesh = as_mesh(trimesh.load(input_path))

    mesh_max = mesh.vertices.max(axis=0)
    mesh_min = mesh.vertices.min(axis=0)
    mesh_center = (mesh_min + mesh_max) / 2
    mesh_scale = mesh_max - mesh_min

    xyz_normed = (mesh.vertices - mesh_center) / mesh_scale.max()

    direction = np.array([1, 0, 0])
    rotation = R.from_rotvec(degree * direction, degrees=True)
    xyz_rotated = rotation.apply(xyz_normed)

    mesh.vertices = xyz_rotated
    mesh.export(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mesh Matching")
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--target_path", "-t", required=True, type=str)
    parser.add_argument("--degree", "-d", required=True, type=int)
    args = parser.parse_args(sys.argv[1:])

    normalize_mesh(args.source_path, args.target_path, args.degree)
    pass
