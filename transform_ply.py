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

# import mesh_to_sdf
import trimesh
import subprocess
import numpy as np
from datetime import datetime
import time

from LOctree import LOctreeA

from transform_utils import *
import torchvision

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(
    dataset,
    opt,
    pipe,
):
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)

    octree = None
    scene = Scene(dataset, gaussians, octree, load_iteration=-1)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    # set random camera
    viewpoint_stack = scene.getTrainCameras().copy()
    # viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
    viewpoint_cam = viewpoint_stack.pop(len(viewpoint_stack) // 4)

    render_path = os.path.join(dataset.model_path, "render_ply")
    os.makedirs(render_path, exist_ok=True)

    image = render(viewpoint_cam, gaussians, pipe, bg)["render"]
    torchvision.utils.save_image(
        image, os.path.join(render_path, "{:.3f}".format(0) + ".png")
    )

    deform_ply_path = os.path.join(dataset.model_path, "deform_ply")
    ply_file_names = os.listdir(deform_ply_path)
    opt.iterations = len(ply_file_names)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Rendering")
    # first_iter += 1
    for iteration in range(first_iter, opt.iterations):
        # Render
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        t = iteration / opt.iterations

        deform_file_path = os.path.join(deform_ply_path, ply_file_names[iteration])
        gaussians.translate_by_ply(deform_file_path)

        image = render(viewpoint_cam, gaussians, pipe, bg)["render"]
        torchvision.utils.save_image(
            image, os.path.join(render_path, ply_file_names[iteration][:-4] + ".png")
        )

        with torch.no_grad():
            # Progress bar
            progress_bar.update()
            if iteration == opt.iterations - 1:
                progress_bar.close()


def prepare_output_and_logger(args):
    # if not args.model_path:
    #     args.model_path = os.path.join(
    #         "./output/", datetime.now().strftime("%m-%d-%H:%M:%S")
    #     )

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


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
