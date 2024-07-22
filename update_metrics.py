import os
import subprocess, sys
import torchvision.transforms.functional as tf
from PIL import Image
from utils.image_utils import psnr
from tqdm import tqdm
import json
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

source_dir = "data/result_files"
metrics = ["FPS", "psnr", "points"]

for metric in metrics:
    source_file = os.path.join(source_dir, f"all_{metric}.json")
    with open(source_file, "r") as f:
        source_dict = json.load(f)

    for dataset_name in os.listdir(source_dir):
        if dataset_name[-5:] == ".json":
            continue

        dataset_dir = os.path.join(source_dir, dataset_name)
        dataset_dict = {}
        for method_name in os.listdir(dataset_dir):
            if not os.path.exists(
                os.path.join(dataset_dir, method_name, f"{metric}.json")
            ):
                continue

            with open(
                os.path.join(dataset_dir, method_name, f"{metric}.json"), "r"
            ) as f:
                json_dict = json.load(f)

            avg = 0.0
            for instance_name in json_dict:
                avg += json_dict[instance_name]
            avg /= len(json_dict)

            dataset_dict[method_name] = avg
        source_dict[dataset_name] = dataset_dict

    if metric == "points":
        for method in os.listdir(dataset_dir):
            if not os.path.exists(os.path.join(dataset_dir, method, f"{metric}.json")):
                continue

            avg = source_dict["Omnizl"][method]
            percentage = avg / source_dict["Omnizl"]["ours"] - 1
            print(f"{method}: +{percentage*100:.2f}%")

    print(source_dict)
    with open(source_file, "w") as f:
        json.dump(source_dict, f)
