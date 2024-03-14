#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, l2_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, logistic_sigmoid
from utils.system_utils import mkdir_p
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from plyfile import PlyData, PlyElement

# import mesh_to_sdf
import trimesh
import subprocess
import numpy as np
from datetime import datetime
import time

from transform_utils import *
import torchvision

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def load_rotation_matrix(path, total_cnt):
    identity_matrix = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    rotation_matrixs = np.expand_dims(identity_matrix, 0).repeat(total_cnt, axis=0)
    with open(path, "r") as f:
        rotation_file = f.read().split("\n")
        for i in range(len(rotation_file) // 4):
            point_index = int(rotation_file[4 * i])
            matrix_str = " ".join(rotation_file[4 * i + 1 : 4 * i + 4])
            rot_matrix = np.fromstring(matrix_str, sep=" ").reshape(3, 3)

            rotation_matrixs[point_index] = rot_matrix
    return rotation_matrixs


def training(
    dataset,
    opt,
    pipe,
):
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.load_ply("data/Omni/dinosaur003/after_deform_info/15k.ply")

    octree = None
    scene = Scene(dataset, gaussians, octree, load_iteration=-1)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    total_cnt = gaussians.get_xyz.shape[0]

    rotation_matrixs = load_rotation_matrix(
        "data/Omni/dinosaur003/after_deform_info/gaussian_rotation_mat_file.txt",
        total_cnt,
    )
    gaussians.translate_by_ply(
        "data/Omni/dinosaur003/after_deform_info/gaussian_new_pos.ply"
    )
    gaussians.rotation_fix(
        torch.ones(total_cnt), torch.from_numpy(rotation_matrixs).float().cuda()
    )

    render_path = os.path.join(dataset.source_path, "render")
    os.makedirs(render_path, exist_ok=True)

    viewpoint_stack = scene.getTrainCameras().copy()
    # viewpoint_cam = viewpoint_stack.pop(6)
    for viewpoint_cam in viewpoint_stack:
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        image = render(viewpoint_cam, gaussians, pipe, bg)["render"]
        torchvision.utils.save_image(
            image,
            os.path.join(
                render_path, "moved_{}".format(viewpoint_cam.image_name) + ".png"
            ),
        )

    # bg = torch.rand((3), device="cuda") if opt.random_background else background
    # image = render(viewpoint_cam, gaussians, pipe, bg)["render"]
    # torchvision.utils.save_image(
    #     image, os.path.join(render_path, "{:.3f}".format(0) + ".png")
    # )

    # image = render(viewpoint_cam, gaussians, pipe, bg)["render"]
    # torchvision.utils.save_image(
    #     image, os.path.join(render_path, "move_{:.3f}".format(1) + ".png")
    # )


if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    with torch.no_grad():
        training(
            lp.extract(args),
            op.extract(args),
            pp.extract(args),
        )

    # All done
    print("Rendering complete.")
