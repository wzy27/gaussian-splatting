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

from LOctree import LOctreeA
from LOTDataset.Dataset import datasetsLOT
from svox import utils
from LOTCorr import Rays
from LOTreeOptGS import InitialFlags, tree_image_eval
import os.path as osp

import time

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Continuous learning rate decay function. Adapted from JaxNeRF

    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.

    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)

    first_epoch = 0
    epoch_bar = tqdm(range(first_epoch, opt.epochs), desc="Overall Training progress")
    first_epoch += 1

    # Octree init
    #########################################################################################
    device = "cuda"  # "cuda" if torch.cuda.is_available() else "cpu"

    octree = LOctreeA.LOTLoad(
        path=os.path.join(dataset.source_path, "LOT_new_table_512.npz"),
        path_d=None,
        load_full=False,
        load_dict=False,
        device=device,
        dtype=torch.float32,
    )
    octree.offset = octree.offset.to("cpu")
    octree.invradius = octree.invradius.to("cpu")
    octree._invalidate()

    FLAGS = InitialFlags()

    octree_vis_dir = osp.splitext(FLAGS.input)[0] + "_render"
    os.makedirs(octree_vis_dir, exist_ok=True)

    dset_train = datasetsLOT[FLAGS.dataset](
        FLAGS.data_dir1,
        split="train",
        device=device,
        factor=1.0,
        n_images=100,
        **utils.build_data_options(FLAGS),
    )

    dset_test = datasetsLOT[FLAGS.dataset](
        FLAGS.data_dir1,
        split="test",
        factor=1.0,
        n_images=3,
        **utils.build_data_options(FLAGS),
    )

    lr_sh_func = get_expon_lr_func(
        FLAGS.lr_sh,
        FLAGS.lr_sh_final,
        FLAGS.lr_sh_delay_steps,
        FLAGS.lr_sh_delay_mult,
        FLAGS.lr_sh_decay_steps,
    )

    lr_sdf_func = get_expon_lr_func(
        FLAGS.lr_sdf,
        FLAGS.lr_sdf_final,
        FLAGS.lr_sdf_delay_steps,
        FLAGS.lr_sdf_delay_mult,
        FLAGS.lr_sdf_decay_steps,
    )

    epoch_id = FLAGS.start_id  # initialization from input

    octree.opt.stop_thresh = 1e-8

    radius = 0.5 / octree.invradius
    center = (1 - 2.0 * octree.offset) * radius
    bbox_corner = center - radius
    bbox_corner.cuda()
    bbox_length = radius * 2
    bbox_length.cuda()
    #########################################################################################

    scene = Scene(dataset, gaussians, octree)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    first_iter = 0
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    first_iter += 1

    for epoch in range(first_epoch, args.epochs + 1):
        # Gaussian Training part
        progress_bar = tqdm(
            range(first_iter, first_iter + opt.iterations), desc="Training progress"
        )
        for iteration in range(first_iter, first_iter + opt.iterations + 1):
            iter_start.record()
            gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            loss_dict = {}
            tree_coords = octree.world2tree(gaussians.get_xyz.cpu().detach()).to("cuda")
            SDF_values = octree.querySDFfromTree(tree_coords)
            # # Loss
            # geometry loss - density to SDF
            opacity_density_scaler = (
                5.0  # TODO: adjust this, change to VolSDF function?
            )
            estimated_opacity = 4 * logistic_sigmoid(
                torch.from_numpy(SDF_values).cuda(), opacity_density_scaler
            ).unsqueeze(-1)
            loss_opacity_l2 = l2_loss(gaussians.get_opacity, estimated_opacity)
            loss_dict["opacity_loss"] = loss_opacity_l2
            # loss_opacity_l2 = lambda_opacity = 0

            # scale loss - smallest scale near 0/some threshold?
            loss_scale = torch.mean(gaussians.get_scaling.min(axis=-1).values)
            loss_dict["scale_loss"] = loss_scale
            # loss_scale = lambda_scale = 0

            # TODO: orientation loss - smallest scale direction with SDF direction
            gaussian_orientation = (
                gaussians.get_normal().squeeze()
            )  # (N, 3), normalized
            SDF_orientation = F.normalize(
                octree.queryNormalFromTree(tree_coords)
            )  # (N, 3), manually normalized
            similarity = torch.abs(
                (gaussian_orientation * SDF_orientation).sum(axis=-1)
            )
            loss_orientation = 1 - torch.mean(similarity)
            loss_dict["orientation_loss"] = loss_orientation
            # loss_orientation = lambda_orientation = 0

            # points near surface loss - SDF near 0
            # loss_point_cloud = torch.mean(SDF_values)
            # lambda_point_cloud = 0.1

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss_dict["image_l1_loss"] = Ll1
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
                1.0 - ssim(image, gt_image)
            )

            loss = (
                loss
                + dataset.lambda_scale * loss_scale
                + dataset.lambda_orientation * loss_orientation
                + dataset.lambda_opacity * loss_opacity_l2
            )
            loss_dict["total_loss"] = loss

            loss.backward()

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                # Log and save
                training_report(
                    tb_writer,
                    iteration,
                    loss_dict,
                    l1_loss,
                    iter_start.elapsed_time(iter_end),
                    testing_iterations,
                    scene,
                    render,
                    (pipe, background),
                )
                if iteration in saving_iterations:
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter],
                    )
                    gaussians.add_densification_stats(
                        viewspace_point_tensor, visibility_filter
                    )

                    if (
                        iteration > opt.densify_from_iter
                        and iteration % opt.densification_interval == 0
                    ):
                        size_threshold = (
                            20 if iteration > opt.opacity_reset_interval else None
                        )
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold,
                            0.005,
                            scene.cameras_extent,
                            size_threshold,
                        )

                    if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter
                    ):
                        gaussians.reset_opacity()

                # Optimizer step
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

                if iteration in checkpoint_iterations:
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save(
                        (gaussians.capture(), iteration),
                        scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                    )

        # Octree Training part
        epoch_id += 1
        dset_train.shuffle_rays()

        epoch_size = dset_train.rays.origins.size(0)
        batches_per_epoch = (epoch_size - 1) // FLAGS.batch_size + 1
        gstep_id_base = epoch_id * batches_per_epoch

        pbar = tqdm(
            enumerate(range(0, epoch_size, FLAGS.batch_size)), total=batches_per_epoch
        )

        tree_coords = octree.world2tree(gaussians.get_xyz.cpu().detach()).to("cuda")
        for iter_id, batch_begin in pbar:
            gstep_id = iter_id + gstep_id_base

            batch_end = min(batch_begin + FLAGS.batch_size, epoch_size)

            batch_origins = dset_train.rays.origins[batch_begin:batch_end]
            batch_dirs = dset_train.rays.dirs[batch_begin:batch_end]

            lr_sh = lr_sh_func(gstep_id)
            lr_sdf = lr_sdf_func(gstep_id)

            if epoch_id >= 6:
                lr_sdf = 5e-5

            im_gt = dset_train.rays.gt[batch_begin:batch_end]

            rays = Rays(batch_origins, batch_dirs)

            im, gsc_loss = octree.VolumeRenderGaussCorr(
                rays, im_gt, tree_coords.contiguous(), scale=1.0
            )

            mse = ((im - im_gt) ** 2).mean()

            psnr = -10.0 * np.log(mse.detach().cpu()) / np.log(10.0)
            # tpsnr += psnr.item()

            if iter_id % 100 == 0:
                pbar.set_postfix(
                    {"PSNR": f"{psnr:.{2}f}", "GSC loss": f"{gsc_loss:.{4}f}"}
                )
            # print("octree_step: ", gstep_id)
            # print("PSNR: ", psnr)
            # print("gsc loss: ", gsc_loss)

            octree.OptimSDF(lr_sdf, beta=0.95)
            octree.OptimSH(lr_sh, beta=0.95)

            if epoch_id < 4:
                octree.Beta.data[0] = 1.0 / (gstep_id / 300.0 + 10.0)
            else:
                octree.Beta.data[0] = 1.0 / ((gstep_id - 4.0 * 12800) / 300.0 + 200.0)

        octree.ExtractGeometry(
            1024,
            bbox_corner,
            bbox_length,
            FLAGS.data_dir1,
            iter_step=gstep_id,
        )

        tree_image_eval(octree, dset_test, 3, octree_vis_dir, device)

        epoch_bar.update()

    pass


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        # args.model_path = os.path.join("./output/", unique_str[0:10])
        args.model_path = os.path.join(
            "./output/", datetime.now().strftime("%m-%d-%H:%M:%S")
        )

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    loss_dict,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
):
    if tb_writer:
        for name, value in loss_dict.items():
            tb_writer.add_scalar(
                "train_loss_patches/{}".format(name), value.item(), iteration
            )
        tb_writer.add_scalar("iter_time", elapsed, iteration)
        tb_writer.add_scalar(
            "total_points", scene.gaussians.get_xyz.shape[0], iteration
        )

    if iteration % 1000 == 0 and iteration < 5000:
        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
            )

    if iteration % 3000 == 0 and iteration >= 5000:
        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
            )

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"],
                        0.0,
                        1.0,
                    )
                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )

        torch.cuda.empty_cache()


if __name__ == "__main__":
    dense_test_iter = [7000, 30000]
    for i in range(0, 5000, 400):
        dense_test_iter.append(i)
    dense_test_iter.sort()

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=dense_test_iter
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
    )

    # All done
    print("\nTraining complete.")
