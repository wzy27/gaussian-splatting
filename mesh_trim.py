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

    return mesh_scale


def normalize_mesh(ours_path, gt_path):
    ours_mesh = as_mesh(trimesh.load(ours_path))
    gt_mesh = as_mesh(trimesh.load(gt_path))

    ours_scale = get_mesh_info(ours_mesh)
    gt_scale = get_mesh_info(gt_mesh)

    print("ours:", ours_scale)
    print("gt:", gt_scale)

    ours_ratio = ours_scale[2] / ours_scale[0]
    gt_ratio = gt_scale[1] / gt_scale[0]

    return ours_ratio / gt_ratio
    # if gt_ratio / ours_ratio > 1.05:


base_dir = "/data/nglm005/zhengyu.wen/final/gaussian-splatting/data"

exp_dirs = [
    "Omnizl",
    # "blender",
]

data_dirs = []

result_dict = {}
for exp_dir in exp_dirs:
    for data_name in os.listdir(os.path.join(base_dir, exp_dir)):
        ours_path = os.path.join(base_dir, exp_dir, data_name, "meshes/00100000.ply")
        if os.path.exists(ours_path):
            data_dirs.append((exp_dir, data_name))

for exp_dir, data_name in sorted(data_dirs):
    load_path = {
        "ours": os.path.join(base_dir, exp_dir, data_name, "meshes/00100000.ply"),
        "gt": os.path.join(base_dir, "gt", data_name, "Scan/Scan.obj"),
    }

    result_dict[data_name] = normalize_mesh(load_path["ours"], load_path["gt"])
    print(data_name, result_dict[data_name])

with open("ratio.json", "w+") as json_file:
    json.dump(result_dict, json_file)

# if __name__ == "__main__":

# parser = argparse.ArgumentParser(description="Mesh Matching")
# parser.add_argument("--source_path", "-s", required=True, type=str)
# parser.add_argument("--target_path", "-t", required=True, type=str)
# args = parser.parse_args(sys.argv[1:])

# normalize_mesh(args.source_path, args.target_path)
