import os
import subprocess, sys
import torchvision.transforms.functional as tf
from PIL import Image
from utils.image_utils import psnr
from tqdm import tqdm
import json
from pathlib import Path

exp_dirs = [
    "/data/nglm005/zhengyu.wen/final/gaussian-splatting/data/Omnizl",
    "/data/nglm005/zhengyu.wen/final/gaussian-splatting/data/blender",
]

data_dirs = []

for exp_dir in exp_dirs:
    for data_name in os.listdir(exp_dir):
        if os.path.exists(os.path.join(exp_dir, data_name, "render_base_test_black")):
            data_dirs.append(os.path.join(exp_dir, data_name))

data_dirs.sort()
print(data_dirs)
print(len(data_dirs))

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["PYTHONPATH"] = "$PYTHONPATH:/data/nglm005/zhengyu.wen/LOTree-zhengyu"


def loadImage(base_dir, name):
    raw = Image.open(os.path.join(base_dir, name))
    return tf.to_tensor(raw).unsqueeze(0)[:, :3, :, :].cuda()


good_dict = {}
for data_dir in data_dirs:
    with open(os.path.join(data_dir, "PSNR_black.json"), "w+") as PSNR_json:
        gt_dir = os.path.join(data_dir, "gt_test_black")
        base_dir = os.path.join(data_dir, "render_base_test_black")
        our_dir = os.path.join(data_dir, "render_test_black")

        result_dict = {}
        for fname in sorted(os.listdir(gt_dir)):
            base = loadImage(base_dir, fname)
            our = loadImage(our_dir, fname)
            gt = loadImage(gt_dir, fname)

            result = {
                "base": "{:.2f}".format(float(psnr(base, gt))),
                "ours": "{:.2f}".format(float(psnr(our, gt))),
            }
            result_dict[fname] = result
            print(result)

            PSNR_delta = float(result["ours"]) - float(result["base"])
            if PSNR_delta > 0 and float(result["base"]) < 35:
                name = Path(data_dir).stem
                good_dict[os.path.join(name, fname)] = PSNR_delta
        json.dump(result_dict, PSNR_json)
        # print(Path(data_dir))
with open("good_PSNR.json", "w+") as good_json:
    json.dump(good_dict, good_json)
