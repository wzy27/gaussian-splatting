import os
import subprocess, sys
import torchvision.transforms.functional as tf
from PIL import Image
from utils.image_utils import psnr
from tqdm import tqdm
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["PYTHONPATH"] = "$PYTHONPATH:/data/nglm005/zhengyu.wen/LOTree-zhengyu"

# with open("data/need_names.json", "r") as f:
with open("data/need_names_test.json", "r") as f:
    name_dict = json.load(f)

# methods = ["2dgs", "gof", "sugar", "neus2", "ours", "voxurf"]  #  "voxurf","2dgs"

methods = ["ours"]  # done: "gof", "sugar", "neus2", "ours"
paths = {
    "gt": "gt.ply",
    "ours": "ours.ply",
    "voxurf": "voxurf.ply",
    "gof": "GOF-matched.ply",
    "sugar": "sugar.ply",
    "neus2": "neus2.ply",
    "2dgs": "2dgs.ply",
}
instances = sorted(name_dict.keys())
# CD_path = "data/result_files/all_CD-as_mesh.json"
CD_path = "data/result_files/test_CD-as_mesh.json"

from chamferDistance import calc_CD


def get_one_CD(arg):
    instance, method = arg
    # print(instance, method)

    with open(CD_path, "r") as f:
        CD_dict = json.load(f)
    if instance in CD_dict:
        instance_dict = CD_dict[instance]
    if instance in CD_dict and method in CD_dict[instance]:
        return

    gt_path = os.path.join("data/meshes", instance, paths["gt"])
    method_path = os.path.join("data/meshes", instance, paths[method])

    print(method, instance)
    result_dict = calc_CD(method_path, gt_path)

    with open(CD_path, "r") as f:
        CD_dict = json.load(f)
    if instance in CD_dict:
        instance_dict = CD_dict[instance]
    instance_dict[method] = result_dict
    CD_dict[instance] = instance_dict

    with open(CD_path, "w") as f:
        json.dump(CD_dict, f)


param_list = []

for instance in instances:
    instance_dict = {}
    for method in methods:
        if method == "gt":
            continue
        with open(CD_path, "r") as f:
            CD_dict = json.load(f)
        if instance in CD_dict:
            instance_dict = CD_dict[instance]
        if instance in CD_dict and method in CD_dict[instance]:
            continue

        # get_one_CD(instance, method)
        param_list.append((instance, method))

print(param_list)

with ProcessPoolExecutor(8) as exe:
    # perform calculations
    results = exe.map(get_one_CD, param_list)
