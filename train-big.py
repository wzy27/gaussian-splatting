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

    instance = os.path.basename(dataset.source_path)
    pretrain_dir = "/data/nglm005/zhengyu.wen/pretrain"
    octree = LOctreeA.LOTLoad(
        path=os.path.join(pretrain_dir, "SDF_512_{}.npz".format(instance)),
        path_d=os.path.join(pretrain_dir, "dict_512_{}.npy".format(instance)),
        load_full=True if dataset.gaussian_smooth else False,
        load_dict=True,
        device=device,
        dtype=torch.float32,
    )
    octree.offset = octree.offset.to("cpu")
    octree.invradius = octree.invradius.to("cpu")
    octree._invalidate()

    FLAGS = InitialFlags(dataset.source_path)

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

    # octree.ExtractGeometry(
    #     512,
    #     bbox_corner,
    #     bbox_length,
    #     dataset.model_path,
    #     iter_step=90000,
    # )

    octree = None
    scene = Scene(dataset, gaussians, octree)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    for idx in range(args.epochs):
        saving_iterations.append(opt.iterations * (idx + 1))
        for cut in range(1, 4):
            testing_iterations.append(opt.iterations * idx + cut * opt.iterations // 3)
    saving_iterations.append(opt.iterations * args.epochs + opt.final_iterations)
    testing_iterations.append(opt.iterations * args.epochs + opt.final_iterations)

    first_iter = 0
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    first_iter += 1
    EMA_PSNR = 20.0

    for epoch in range(first_epoch, args.epochs + 1):
        # Gaussian Training part
        progress_bar = tqdm(
            range(first_iter, first_iter + opt.iterations), desc="Training progress"
        )
        for iteration in range(first_iter, first_iter + opt.iterations + 1):
            iter_start.record()
            gaussians.update_learning_rate(iteration)

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
            # inner - negative, outer - positive
            SDF_values = torch.from_numpy(octree.querySDFfromTree(tree_coords)).cuda()
            # # Loss TODO: increase as traning proceeds
            # geometry loss - density to SDF
            near_threshold = 0.1
            near_mask = torch.logical_or(
                SDF_values < -near_threshold, SDF_values > near_threshold
            )

            opacity_density_scaler = 5.0
            estimated_opacity = 4 * logistic_sigmoid(
                SDF_values, opacity_density_scaler
            ).unsqueeze(-1)
            truncated_opacity = torch.where(near_mask, 0.0, 1.0).unsqueeze(-1)
            estimated_opacity = torch.min(estimated_opacity, truncated_opacity)

            loss_opacity_l2 = l2_loss(gaussians.get_opacity, estimated_opacity)
            loss_dict["opacity_loss"] = loss_opacity_l2

            # scale loss - smallest scale near 0/some threshold?
            if dataset.lambda_scale > 0:
                loss_scale = torch.mean(gaussians.get_scaling.min(axis=-1).values)
                loss_dict["scale_loss"] = loss_scale
            else:
                loss_scale = 0

            # orientation loss - smallest scale direction with SDF direction
            if dataset.lambda_opacity > 0:
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
            else:
                loss_orientation = 0

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
                    extract_threshold = near_threshold
                    extract_mask = torch.logical_and(
                        SDF_values >= -extract_threshold,
                        SDF_values <= extract_threshold,
                    )

                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)
                    gaussians.extract_by_mask(
                        extract_mask,
                        os.path.join(
                            scene.model_path,
                            "opacity/it_{}_{:.6f}.ply".format(
                                iteration, torch.var(SDF_values)
                            ),
                        ),
                    )
                    print(
                        "Keep ratio: {:.4f}".format(
                            extract_mask.sum() / extract_mask.shape[0]
                        )
                    )

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
                if iteration < args.epochs * opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

        first_iter += opt.iterations

        # Octree Training part
        epoch_id += 1
        dset_train.shuffle_rays()

        epoch_size = dset_train.rays.origins.size(0) // 2  # total #rays used in a epoch
        batches_per_epoch = (epoch_size - 1) // FLAGS.batch_size + 1
        gstep_id_base = 76800 + batches_per_epoch * (epoch - 1)

        pbar = tqdm(
            enumerate(range(0, epoch_size, FLAGS.batch_size)), total=batches_per_epoch
        )

        extract_threshold = near_threshold
        extract_mask = torch.logical_and(
            SDF_values >= -extract_threshold,
            SDF_values <= extract_threshold,
        )

        tree_coords = octree.world2tree(gaussians.get_xyz.cpu().detach()).to("cuda")
        tree_coords = tree_coords[extract_mask].contiguous()

        if dataset.hessian_eikonal:
            print("Begin gaussian sampling for hessian")

            octree.SampleGaussianPoints(
                gaussian_sigma=0.1, max_n_samples=64, gauss_sampling=False
            )
            point_count = 1.5e6
            sparse_frac_hess_eiko = point_count / float(
                octree.GaussianSamplePoints.size(0)
            )

            octree.opt.hess_step = 1.0 / (2**9)

            # gaussian_sigma = 0.3
            # max_n_samples = 256
            # octree.SampleGaussianPoints(
            #     gaussian_sigma=gaussian_sigma, max_n_samples=max_n_samples
            # )

            # octree.opt.hess_step = 0.0019

        for iter_id, batch_begin in pbar:
            gstep_id = iter_id + gstep_id_base

            batch_end = min(batch_begin + FLAGS.batch_size, epoch_size)

            batch_origins = dset_train.rays.origins[batch_begin:batch_end]
            batch_dirs = dset_train.rays.dirs[batch_begin:batch_end]

            lr_sh = lr_sh_func(gstep_id)
            lr_sdf = lr_sdf_func(gstep_id)

            # if epoch_id >= 6:
            #     lr_sdf = 5e-5

            im_gt = dset_train.rays.gt[batch_begin:batch_end]

            rays = Rays(batch_origins, batch_dirs)

            scale_gauss = 1  # TODO: gaussian smoother lambda
            sparse_frac_gauss_smooth = 0.1
            im, gauss_smooth_loss = octree.VolumeRenderWOGaussVolSDF(
                rays,
                im_gt,
                scale=scale_gauss,
                ksize_grad=3,
                sparse_frac=sparse_frac_gauss_smooth,
                gauss_grad_loss=dataset.gaussian_smooth,
            )

            if dataset.hessian_eikonal:
                scale_hess = 1e-12
                scale_eiko = 1e-6
                (
                    hessian_loss,
                    eikonal_loss,
                    _,
                    _,
                    _,
                ) = octree.VolumePointsHessEikon(
                    hessian_on=True,
                    eikonal_on=True,
                    scale_h=scale_hess,
                    scale_e=scale_eiko,
                    eikon_thres_min=0.8,
                    eikon_thres_max=1.5,
                    sparse_frac=sparse_frac_hess_eiko,
                )

            if dataset.gauss_splat_corr:
                scale_gsc = 0.3  # TODO: gaussian-SDF loss lambda
                SDF_thre = -0.05  # avg inner scale of points
                gsc_loss = octree.VolumeRenderGaussCorr(
                    tree_coords, scale_gsc, SDF_thre
                )

            mse = ((im - im_gt) ** 2).mean()
            psnr = -10.0 * np.log(mse.detach().cpu()) / np.log(10.0)
            EMA_PSNR = 0.99 * EMA_PSNR + 0.01 * psnr

            if iter_id % 100 == 0:
                if dataset.gaussian_smooth:
                    pbar.set_postfix(
                        {
                            # "PSNR": f"{EMA_PSNR:.{2}f}",
                            "Gauss Smooth Loss": f"{gauss_smooth_loss:.{4}f}",
                        }
                    )

                if dataset.hessian_eikonal:
                    pbar.set_postfix(
                        {
                            # "PSNR": f"{psnr:.{2}f}",
                            "Hessian Loss": f"{hessian_loss:.{4}f}",
                            "Eikonal Loss": f"{eikonal_loss:.{4}f}",
                        }
                    )

                if dataset.gauss_splat_corr:
                    pbar.set_postfix(
                        {"PSNR": f"{EMA_PSNR:.{2}f}", "GSC Loss": f"{gsc_loss:.{4}f}"}
                    )

            octree.OptimSDF(lr_sdf, beta=0.95)
            octree.OptimSH(lr_sh, beta=0.95)

            if epoch_id < 4:
                octree.Beta.data[0] = 1.0 / (gstep_id / 300.0 + 10.0)
            else:
                octree.Beta.data[0] = 1.0 / ((gstep_id - 4.0 * 12800) / 300.0 + 200.0)
                # octree.Beta.data[0] = 1.0 / ((gstep_id - 4.0 * epoch_size) / 300.0 + 200.0)

        octree.ExtractGeometry(
            512,
            bbox_corner,
            bbox_length,
            dataset.model_path,
            iter_step=gstep_id + 1,
        )

        tree_image_eval(octree, dset_test, 3, octree_vis_dir, device)
        epoch_bar.update()

    pass

    octree.LOTSave(
        path=os.path.join(dataset.model_path, "SDF_iter_{}.npz".format(first_iter)),
        path_d=os.path.join(dataset.model_path, "dict_iter_{}.npy".format(first_iter)),
        out_full=True,
    )

    # final stage, add gaussians back
    # Gaussian Training part
    progress_bar = tqdm(
        range(first_iter, first_iter + opt.final_iterations), desc="Training progress"
    )
    for iteration in range(first_iter, first_iter + opt.final_iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        loss_dict = {}
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss_dict["image_l1_loss"] = Ll1
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0 - ssim(image, gt_image)
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
                extract_threshold = near_threshold
                extract_mask = torch.logical_and(
                    SDF_values >= -extract_threshold,
                    SDF_values <= extract_threshold,
                )

                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < first_iter + 5000:
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
            if iteration < first_iter + opt.final_iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)


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

    # Report test and samples of training set
    if iteration in testing_iterations:
        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
            )
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
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    # args.save_iterations.append(args.iterations)

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
        args.debug_from,
    )

    # All done
    print("\nTraining complete.")
