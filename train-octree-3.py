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
import gc
import imageio

from LOctree import LOctreeA, CreateNewTreeBasedPast
from LOTDataset.Dataset import datasetsLOT
from svox import utils
from LOTCorr import Rays
from LOTreeOptGS import InitialFlags, tree_image_eval
from LOTRenderer import LOTRenderA

from LOTCorr import RenderOptions
from svox2 import defs

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

FLAGS = InitialFlags("")


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


def OctreeSDFOpt(
    root_path,
    model_name,
    point_path,
    train_at_beginning,
    train_with_gsp,
    scale_init_hess,
    scale_init_lapla,
):
    device = "cuda"

    print("LOTreeA load")

    c_path = root_path + "/" + model_name

    train_at_beginning = train_at_beginning
    train_with_gsp = train_with_gsp

    if train_at_beginning:
        t = LOctreeA(
            N=2,
            data_dim=28,
            basis_dim=9,
            init_refine=0,
            init_reserve=50000,
            geom_resize_fact=1.0,
            center=torch.Tensor([0.0, 0.0, 0.0]),
            radius=torch.Tensor([1.0, 1.0, 1.0]),
            depth_limit=11,
            data_format=f"SH{9}",
            device=device,
        )
        t.InitializeCorners()

        for i in range(5):
            t.RefineCorners()

        t.InitializeOctree()

        t.SampleGaussianPoints(
            gaussian_sigma=0.3, max_n_samples=256, gauss_sampling=False
        )
        point_count = 1.0e6
        sparse_frac_points = point_count / float(t.GaussianSamplePoints.size(0))

        epoch_id = -1
        gstep_id_base = 0

        bbox_corner = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
        bbox_length = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    else:
        path_octree = (
            "/data/nglm005/zhengyu.wen/pretrain/" + "SDF_128_" + model_name + ".npz"
        )
        path_octree_dic = (
            "/data/nglm005/zhengyu.wen/pretrain/" + "SDF_128_dic_" + model_name + ".npy"
        )

        t = LOctreeA.LOTLoad(
            path=path_octree,
            path_d=path_octree_dic,
            load_full=True,
            load_dict=True,
            device=device,
            dtype=torch.float32,
        )

        t.density_rms = None
        t.sh_rms = None
        t.sdf_rms = None
        t.beta_rms = None
        t.learns_rms = None
        t.opt = RenderOptions()
        t.basis_type = defs.BASIS_TYPE_SH
        t.offset = t.offset.to("cpu")
        t.invradius = t.invradius.to("cpu")

        t._invalidate()

        epoch_id = 1
        gstep_id_base = 12800 * (epoch_id + 1)

        radius = 0.5 / t.invradius
        center = (1 - 2.0 * t.offset) * radius
        bbox_corner = center - radius
        bbox_corner.cuda()
        bbox_length = radius * 2
        bbox_length.cuda()

        t.stop_thresh = 1e-8
        t.opt.hess_step = 1.0 / (2.0**7)

        if train_with_gsp:
            gsp_points = t.LoadGSP(gaussian_points_path=point_path)

    vis_dir = c_path + "/tree_render"

    dset_train = datasetsLOT[FLAGS.dataset](
        c_path,
        split="train",
        device=device,
        factor=1.0,
        n_images=100,
        **utils.build_data_options(FLAGS),
    )

    dset_test = datasetsLOT[FLAGS.dataset](
        c_path,
        split="test",
        factor=1.0,
        n_images=3,
        **utils.build_data_options(FLAGS),
    )

    os.makedirs(vis_dir, exist_ok=True)

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

    while True:
        epoch_id += 1

        dset_train.shuffle_rays()

        epoch_size = dset_train.rays.origins.size(0)
        batches_per_epoch = (epoch_size - 1) // FLAGS.batch_size + 1

        def train_step():
            print("Train step")

            tpsnr = 0.0
            pbar = tqdm(
                enumerate(range(0, epoch_size, FLAGS.batch_size)),
                total=batches_per_epoch,
            )
            for iter_id, batch_begin in pbar:
                gstep_id = iter_id + gstep_id_base

                lr_sh = lr_sh_func(gstep_id)
                lr_sdf = lr_sdf_func(gstep_id)

                batch_end = min(batch_begin + FLAGS.batch_size, epoch_size)
                batch_origins = dset_train.rays.origins[batch_begin:batch_end]
                batch_dirs = dset_train.rays.dirs[batch_begin:batch_end]

                im_gt = dset_train.rays.gt[batch_begin:batch_end]

                rays = Rays(batch_origins, batch_dirs)

                if epoch_id < 1:
                    im, smooth_loss = t.VolumeRenderWOGaussVolSDF(
                        rays,
                        im_gt,
                        scale=2e-4,
                        ksize_grad=3,
                        sparse_frac=1.0,
                        gauss_grad_loss=False,
                    )

                    (
                        hessian_loss,
                        eikonal_loss,
                        laplacian_loss,
                        dirichlet_loss,
                        viscosity_loss,
                    ) = t.VolumePointsHessEikon(
                        hessian_on=True,
                        eikonal_on=True,
                        laplacian_on=True,
                        dirichlet_on=False,
                        viscosity_on=False,
                        scale_h=scale_init_hess,
                        scale_e=1e-6,
                        scale_l=scale_init_lapla,
                        scale_d=5e-2,
                        scale_v=5e-5,
                        eikon_thres_min=0.8,
                        eikon_thres_max=1.2,
                        viscosity_thres=0.0,
                        sparse_frac=sparse_frac_points,
                    )

                else:
                    if epoch_id < 3:
                        gauss_grad_loss = False
                        scale = 2e-4
                        sparse_frac = 1.0
                    elif epoch_id < 6:
                        gauss_grad_loss = False
                        scale = 2e-4
                        sparse_frac = 0.1
                    else:
                        gauss_grad_loss = False
                        scale = 2e-4
                        sparse_frac = 0.1

                    im, smooth_loss = t.VolumeRenderWOGaussVolSDF(
                        rays,
                        im_gt,
                        scale=scale,
                        ksize_grad=3,
                        sparse_frac=sparse_frac,
                        gauss_grad_loss=gauss_grad_loss,
                    )

                    if epoch_id < 2:
                        scale_h = 1e-9
                        scale_e = 1e-6
                    elif epoch_id < 3:
                        scale_h = 1e-9
                        scale_e = 1e-6
                    elif epoch_id < 4:
                        scale_h = 1e-10
                        scale_e = 1e-6
                    elif epoch_id < 5:
                        scale_h = 1e-11
                        scale_e = 1e-6
                    elif epoch_id < 6:
                        scale_h = 1e-12
                        scale_e = 1e-6
                    elif epoch_id < 8:
                        scale_h = 1e-13
                        scale_e = 1e-6
                    (
                        hessian_loss,
                        eikonal_loss,
                        laplacian_loss,
                        dirichlet_loss,
                        viscosity_loss,
                    ) = t.VolumePointsHessEikon(
                        hessian_on=True,
                        eikonal_on=True,
                        laplacian_on=False,
                        dirichlet_on=False,
                        viscosity_on=False,
                        scale_h=scale_h,
                        scale_e=scale_e,
                        scale_l=5e-8,
                        scale_d=5e-2,
                        scale_v=5e-5,
                        eikon_thres_min=0.8,
                        eikon_thres_max=1.2,
                        viscosity_thres=0.0,
                        sparse_frac=sparse_frac_points,
                    )

                    if train_with_gsp:
                        if epoch_id < 3:
                            scale_gsc = 1e-1
                            scale_gsc_b = 1e-7
                        elif epoch_id < 4:
                            scale_gsc = 1e-2
                            scale_gsc_b = 1e-8
                        elif epoch_id < 6:
                            scale_gsc = 1e-3
                            scale_gsc_b = 1e-8
                        elif epoch_id < 7:
                            scale_gsc = 1e-3
                            scale_gsc_b = 1e-9
                        elif epoch_id < 8:
                            scale_gsc = 1e-3
                            scale_gsc_b = 1e-9
                        gsc_loss = t.VolumeRenderGaussCorr(
                            gsp_points,
                            only_above=True,
                            scale=scale_gsc,
                            scale_below=scale_gsc_b,
                            threshold=-0.01,
                            above_threshold=-0.01,
                            below_threshold=-0.007,
                        )

                mse = ((im - im_gt) ** 2).mean()

                psnr = -10.0 * np.log(mse.detach().cpu()) / np.log(10.0)
                print("step: ", gstep_id)
                print("psnr: ", psnr)
                tpsnr += psnr.item()

                if epoch_id < 1:
                    print("hessian loss: ", hessian_loss)
                    print("eikonal loss: ", eikonal_loss)
                else:
                    print("hessian loss: ", hessian_loss)
                    print("eikonal loss: ", eikonal_loss)

                if train_with_gsp:
                    print("gsc loss: ", gsc_loss)

                if epoch_id < 6:
                    if epoch_id < 2:
                        scaling_sh = 1e-3
                    elif epoch_id < 6:
                        scaling_sh = 1e-5
                    if FLAGS.lambda_tv_sh > 0.0:
                        t.AddTVThirdordColorGrad(
                            t.CornerSH.grad,
                            scaling=scaling_sh,
                            sparse_frac=FLAGS.tv_sparsity,
                            contiguous=True,
                        )

                t.OptimSDF(lr_sdf, beta=FLAGS.rms_beta)
                t.OptimSH(lr_sh, beta=FLAGS.rms_beta)
                # t.OptimBeta(lr_beta, beta=FLAGS.rms_beta)

                if epoch_id < 6:
                    t.Beta.data[0] = 1.0 / (gstep_id / 300.0 + 10.0)
                else:
                    t.Beta.data[0] = 1.0 / ((gstep_id - 4.0 * 12800) / 300.0 + 200.0)

                # print('Beta: ', t.Beta.data[0])

                if gstep_id % 10000 == 0 and gstep_id != 0:
                    test_step()
                    gc.collect()

                if gstep_id % 10000 == 0:
                    mesh_dir = c_path

                    if epoch_id >= 6:
                        resolution = 1024
                    else:
                        resolution = 512

                    t.ExtractGeometry(
                        resolution,
                        bbox_corner,
                        bbox_length,
                        mesh_dir,
                        iter_step=gstep_id,
                    )

                    gc.collect()

            # print('** train_psnr', tpsnr)

        def test_step():
            with torch.no_grad():
                if epoch_id < 0:
                    t.RenderGaussSDFIsolines(z_pos=0.5, vis_dir=vis_dir)
                else:
                    tree_rad = 0.5 / t.invradius
                    t.RenderSDFIsolines(
                        z_pos=0.5, radius=tree_rad, vis_dir=vis_dir, epoch_id=epoch_id
                    )

                # t.offset = t.offset.to(device)
                # t.invradius = t.invradius.to(device)

                N_IMGS_TO_EVAL = 3
                img_eval_interval = dset_test.n_images // N_IMGS_TO_EVAL
                img_ids = range(0, dset_test.n_images, img_eval_interval)

                for i, img_id in tqdm(enumerate(img_ids), total=len(img_ids)):
                    c2w = dset_test.c2w[img_id].to(device=device)

                    width = dset_test.get_image_size(img_id)[1]
                    height = dset_test.get_image_size(img_id)[0]

                    rays = LOTRenderA.GenerateRaysByCamera(
                        c2w,
                        dset_test.intrins.get("fx", img_id),
                        dset_test.intrins.get("fy", img_id),
                        dset_test.intrins.get("cx", img_id),
                        dset_test.intrins.get("cy", img_id),
                        width,
                        height,
                    )

                    rays.origins.view(width * height, -1)
                    rays.dirs.view(width * height, -1)

                    rays_r = Rays(rays.origins, rays.dirs)

                    im_gt = dset_test.gt[img_id].to(device="cpu")
                    im_gt_ten = im_gt.view(width * height, -1)

                    im = t.VolumeRenderVolSDFTest(rays_r)

                    im = im.cpu().clamp_(0.0, 1.0)

                    mse = ((im - im_gt_ten) ** 2).mean()
                    psnr = -10.0 * np.log(mse) / np.log(10.0)
                    print("psnr: ", psnr)

                    # im = im.cpu().clamp_(0.0, 1.0)

                    # mse = ((im - im_gt_ten) ** 2).mean()
                    # psnr = -10.0 * np.log(mse) / np.log(10.0)
                    # print('psnr: ', psnr)

                    im = im.view(height, width, -1)
                    vis = torch.cat((im_gt, im), dim=1)
                    vis = (vis * 255).numpy().astype(np.uint8)
                    imageio.imwrite(f"{vis_dir}/{i:04}_{img_id:04}.png", vis)

        def RefineStep(
            sdf_threshold,
            geo_threshold,
            sdf_offset,
            n_samples,
            com_gauss,
            dilate_times=1,
            com_const_geo=True,
        ):
            with torch.no_grad():
                t.AdaptiveResolutionGeoVolSDF(
                    n_samples=n_samples,
                    sdf_threshold=sdf_threshold,
                    geo_threshold=geo_threshold,
                    sdf_offset=sdf_offset,
                    is_abs=True,
                    dilate_times=dilate_times,
                    com_gauss=com_gauss,
                    com_const_geo=com_const_geo,
                )

        if (
            epoch_id == 1
            or epoch_id == 2
            or epoch_id == 4
            or epoch_id == 6
            or epoch_id == 8
        ):
            if epoch_id == 1:
                # t.CornerSDF.data = t.CornerGaussSDF.data

                new_t, new_bbox_corner, new_bbox_length = CreateNewTreeBasedPast(
                    past_tree=t,
                    refine_time=1,
                    n_samples=512,
                    sdf_thresh=0.0,
                    geo_sdf_thresh=0.2,
                    sdf_thresh_offset=0.0,
                    dilate_times=1,
                    device=device,
                )
                del t
                t = new_t

                tree_rad = 0.5 / t.invradius
                t.RenderSDFIsolines(
                    z_pos=0.5, radius=tree_rad, vis_dir=vis_dir, epoch_id=epoch_id
                )

                bbox_corner = new_bbox_corner
                bbox_length = new_bbox_length

                t.SampleGaussianPoints(
                    gaussian_sigma=0.3, max_n_samples=256, gauss_sampling=False
                )
                point_count = 1.0e6
                sparse_frac_points = point_count / float(t.GaussianSamplePoints.size(0))

                # t.opt.cube_thresh *= 2.0
                t.opt.stop_thresh = 1e-8
                t.opt.hess_step /= 2.0
                t.opt.hess_sdf_thresh = 0.2
                gc.collect()
                # RefineStep(sdf_threshold=1e-1, loss_threshold=0.0)
                # t.VolumeRenderVolSDFRecord(dset_train=dset_train)
            elif epoch_id == 2:
                if not train_with_gsp:
                    path_octree = (
                        "/data/nglm005/zhengyu.wen/pretrain/"
                        + "SDF_128_"
                        + model_name
                        + ".npz"
                    )
                    path_octree_dic = (
                        "/data/nglm005/zhengyu.wen/pretrain/"
                        + "SDF_128_dic_"
                        + model_name
                        + ".npy"
                    )

                    t.LOTSave(
                        path=path_octree,
                        path_d=path_octree_dic,
                        out_full=True,
                    )

                    break

                t.SampleGaussianPoints(
                    gaussian_sigma=0.1, max_n_samples=128, gauss_sampling=False
                )
                point_count = 1.5e6
                sparse_frac_points = point_count / float(t.GaussianSamplePoints.size(0))

                gc.collect()
            elif epoch_id == 4:
                # t.LOTSave(
                #     path="/data/nglm005/zhengyu.wen/data/Omni/sofa010/LOT_new_sofa_128_gsp_full.npz",
                #     path_d="/data/nglm005/zhengyu.wen/data/Omni/sofa010/LOT_new_sofa_128_gsp_dic_full.npy",
                #     out_full=True,
                # )

                RefineStep(
                    sdf_threshold=0.1,
                    geo_threshold=0.1,
                    sdf_offset=0e-1,
                    n_samples=256,
                    com_gauss=False,
                )

                # t.opt.cube_thresh *= 2.0
                # test_step()
                t.opt.hess_step /= 2.0

                t.SampleGaussianPoints(
                    gaussian_sigma=0.1, max_n_samples=128, gauss_sampling=False
                )
                point_count = 1.5e6
                sparse_frac_points = point_count / float(t.GaussianSamplePoints.size(0))

                # train_with_gsp = False

                gc.collect()
            elif epoch_id == 6:
                # if train_at_beginning:
                # t.LOTSave(
                #     path="/data/nglm005/zhengyu.wen/data/Omni/sofa010/LOT_new_sofa_256_gsp.npz",
                #     path_d="/data/nglm005/zhengyu.wen/data/Omni/sofa010/LOT_new_sofa_256_gsp_dic.npy",
                #     out_full=False,
                # )
                # t.LOTSave(
                #     path="/data/nglm005/zhengyu.wen/data/Omni/sofa010/LOT_new_sofa_256_full.npz",
                #     path_d="/data/nglm005/zhengyu.wen/data/Omni/sofa010/LOT_new_sofa_256_dic_full.npy",
                #     out_full=True,
                # )

                RefineStep(
                    sdf_threshold=0.05,
                    geo_threshold=0.05,
                    sdf_offset=0e-1,
                    n_samples=256,
                    com_gauss=False,
                    com_const_geo=False,
                )

                t.opt.hess_step /= 2.0

                t.SampleGaussianPoints(
                    gaussian_sigma=0.1, max_n_samples=64, gauss_sampling=False
                )
                point_count = 1.5e6
                sparse_frac_points = point_count / float(t.GaussianSamplePoints.size(0))

                gc.collect()
            elif epoch_id == 8:
                path_octree = (
                    c_path + "/" + model_name + "/" + "SDF_512_" + model_name + ".npz"
                )
                path_octree_dic = (
                    c_path
                    + "/"
                    + model_name
                    + "/"
                    + "SDF_512_dic_"
                    + model_name
                    + ".npy"
                )
                t.LOTSave(
                    path=path_octree,
                    path_d=path_octree_dic,
                    out_full=False,
                )

                gc.collect()

                break

        train_step()
        print(torch.cuda.memory_allocated())
        print(torch.cuda.max_memory_allocated())
        gc.collect()
        gstep_id_base += 12800


def training(
    dataset,
    root_path,
    model_name,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
):
    path_octree = (
        "/data/nglm005/zhengyu.wen/pretrain/" + "SDF_128_" + model_name + ".npz"
    )
    octree = LOctreeA.LOTLoad(
        path=path_octree,
        path_d=None,
        load_full=False,
        load_dict=False,
        device="cuda",
        dtype=torch.float32,
    )
    octree.offset = octree.offset.to("cpu")
    octree.invradius = octree.invradius.to("cpu")
    octree._invalidate()
    # octree = None

    radius = 0.5 / octree.invradius
    center = (1 - 2.0 * octree.offset) * radius
    bbox_corner = center - radius
    bbox_corner.cuda()
    bbox_length = radius * 2
    bbox_length.cuda()

    mesh_path = root_path + "/" + model_name

    octree.ExtractGeometry(
        512,
        bbox_corner,
        bbox_length,
        mesh_path,
        iter_step=666666,
    )

    dataset.source_path = mesh_path

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians, octree)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        # s1 = time.time()
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

        # s2 = time.time()
        loss_dict = {}
        tree_coords = octree.world2tree(gaussians.get_xyz.cpu().detach()).to("cuda")
        SDF_values = torch.from_numpy(octree.querySDFfromTree(tree_coords)).cuda()
        # s3 = time.time()
        # # Loss
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
        loss_scale = torch.mean(gaussians.get_scaling.min(axis=-1).values)
        loss_dict["scale_loss"] = loss_scale

        # TODO: orientation loss - smallest scale direction with SDF direction
        gaussian_orientation = gaussians.get_normal().squeeze()  # (N, 3), normalized
        SDF_orientation = F.normalize(
            octree.queryNormalFromTree(tree_coords)
        )  # (N, 3), manually normalized
        similarity = torch.abs((gaussian_orientation * SDF_orientation).sum(axis=-1))
        loss_orientation = 1 - torch.mean(similarity)
        loss_dict["orientation_loss"] = loss_orientation

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

        # s4 = time.time()
        # print("time: {}, {}, {}".format(s2 - s1, s3 - s1, s4 - s1))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                SDF_avg = torch.mean(SDF_values)
                SDF_var = torch.var(SDF_values)
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{7}f}",
                        "SDF_avg": f"{SDF_avg:.{7}f}",
                        "SDF_var": f"{SDF_var:.{7}f}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer,
                iteration,
                loss_dict,
                SDF_values,
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
                    SDF_values >= -extract_threshold, SDF_values <= extract_threshold
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
            if iteration < opt.densify_until_iter:  # TODO: prune by SDF?
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
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

    gc.collect()

    point_path = scene.model_path + "/point_cloud" + "/iteration_7000/point_cloud.ply"

    return point_path


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
    sdf_values,
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

        if iteration % 3000 == 0:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
            )
            tb_writer.add_histogram("scene/SDF_histogram", sdf_values, iteration)

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


def GaussianOpt(root_path, model_name):
    dense_test_iter = [7000, 15000]
    # for i in range(0, 7000, 1000):
    #     dense_test_iter.append(i)
    # dense_test_iter.sort()

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
        "--save_iterations", nargs="+", type=int, default=dense_test_iter
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

    # lp._source_path = root_path + "/" + model_name

    point_path = training(
        lp.extract(args),
        root_path,
        model_name,
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
    )

    return point_path


if __name__ == "__main__":
    root_path = "/data/nglm005/zhengyu.wen/data/Omnizl"
    model_names = []

    model_names.append("rice001")
    model_names.append("rice002")
    model_names.append("rice003")
    model_names.append("rice006")
    model_names.append("rice008")

    scale_init_laplas = []
    scale_init_hesses = []
    for i in range(len(model_names)):
        if model_names[i] == "rice001":
            scale_init_laplas.append(1e-7)
            scale_init_hesses.append(1e-7)
        else:
            scale_init_laplas.append(1e-9)
            scale_init_hesses.append(1e-8)

    for i in range(len(model_names)):
        OctreeSDFOpt(
            root_path=root_path,
            model_name=model_names[i],
            point_path=None,
            train_at_beginning=True,
            train_with_gsp=False,
            scale_init_hess=1e-8,
            scale_init_lapla=scale_init_laplas[i],
        )

        point_path = GaussianOpt(root_path=root_path, model_name=model_names[i])

        OctreeSDFOpt(
            root_path=root_path,
            model_name=model_names[i],
            point_path=point_path,
            train_at_beginning=False,
            train_with_gsp=True,
            scale_init_hess=1e-8,
            scale_init_lapla=1e-9,
        )

    # dense_test_iter = [7000, 15000]
    # for i in range(0, 7000, 1000):
    #     dense_test_iter.append(i)
    # dense_test_iter.sort()

    # # Set up command line argument parser
    # parser = ArgumentParser(description="Training script parameters")
    # lp = ModelParams(parser)
    # op = OptimizationParams(parser)
    # pp = PipelineParams(parser)
    # parser.add_argument("--ip", type=str, default="127.0.0.1")
    # parser.add_argument("--port", type=int, default=6009)
    # parser.add_argument("--debug_from", type=int, default=-1)
    # parser.add_argument("--detect_anomaly", action="store_true", default=False)
    # parser.add_argument(
    #     "--test_iterations", nargs="+", type=int, default=dense_test_iter
    # )
    # parser.add_argument(
    #     "--save_iterations", nargs="+", type=int, default=dense_test_iter
    # )
    # parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    # parser.add_argument("--start_checkpoint", type=str, default=None)
    # args = parser.parse_args(sys.argv[1:])
    # args.save_iterations.append(args.iterations)

    # print("Optimizing " + args.model_path)

    # # Initialize system state (RNG)
    # safe_state(args.quiet)

    # torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # training(
    #     lp.extract(args),
    #     op.extract(args),
    #     pp.extract(args),
    #     args.test_iterations,
    #     args.save_iterations,
    #     args.checkpoint_iterations,
    #     args.start_checkpoint,
    #     args.debug_from,
    # )

    # # All done
    # print("\nTraining complete.")
