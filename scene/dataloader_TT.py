import os
import glob
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
from pathlib import Path
from PIL import Image
from scene.dataset_readers import CameraInfo
from utils.graphics_utils import focal2fov


def loadTTImages(basedir, split):
    pose_paths = sorted(
        glob.glob(os.path.join(basedir, "pose_{}".format(split), "*txt"))
    )
    img_paths = sorted(glob.glob(os.path.join(basedir, "{}".format(split), "*png")))

    all_poses = []
    # all_imgs = []
    for i, (pose_path, rgb_path) in enumerate(zip(pose_paths, img_paths)):
        all_poses.append(np.loadtxt(pose_path).astype(np.float32))
        # all_imgs.append((imageio.imread(rgb_path) / 255.0).astype(np.float32))

    # imgs = np.stack(all_imgs, 0)
    poses = np.stack(all_poses, 0)
    return poses, img_paths


def createTTCameraInfo(poses, img_paths, focal, white_background=True):
    cam_infos = []

    for idx, pose, img_path in enumerate(zip(poses, img_paths)):
        # cam_name = os.path.join(path, frame["file_path"] + extension)

        # NeRF 'transform_matrix' is a camera-to-world transform
        # c2w = np.array(frame["transform_matrix"])
        c2w = pose
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(
            w2c[:3, :3]
        )  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_name = Path(img_path).stem
        image = Image.open(img_path)

        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (
            1 - norm_data[:, :, 3:4]
        )
        image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

        # fovy = focal2fov(intrinsics[0, 0], image.size[1])
        FovY = focal2fov(focal, image.size[1])
        FovX = focal2fov(focal, image.size[0])

        cam_infos.append(
            CameraInfo(
                uid=idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_path=img_path,
                image_name=image_name,
                width=image.size[0],
                height=image.size[1],
            )
        )
    return cam_infos


def readTTCameraInfo(basedir, white_background=True):
    train_poses, train_img_paths = loadTTImages(basedir, "train")
    test_poses, test_img_paths = loadTTImages(basedir, "test")

    path_intrinsics = os.path.join(basedir, "intrinsics.txt")
    K = np.loadtxt(path_intrinsics)
    focal = float(K[0, 0])

    train_cam_infos = createTTCameraInfo(
        train_poses, train_img_paths, focal, white_background
    )
    test_cam_infos = createTTCameraInfo(
        test_poses, test_img_paths, focal, white_background
    )

    return train_cam_infos, test_cam_infos
