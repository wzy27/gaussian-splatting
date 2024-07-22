import os, subprocess
import json

name = "table_019"
# dataset = "blender"
# if name[-1].isdight():
dataset = "Omnizl"

methods = ["voxurf", "neus2", "sugar", "2dgs", "gof", "ours"]  # voxurf
output_str = ""
for method in methods:
    with open(f"data/result_files/{dataset}/{method}/psnr.json", "r") as f:
        psnr_dict = json.load(f)
    psnr = psnr_dict[name]
    with open(f"data/result_files/{dataset}/{method}/FPS.json", "r") as f:
        fps_dict = json.load(f)
    fps = fps_dict[name]

    points_path = f"data/result_files/{dataset}/{method}/points.json"
    points = "-"
    result_str = f"& {psnr:.2f}, {fps:.2f}, - "
    if os.path.exists(points_path):
        with open(f"data/result_files/{dataset}/{method}/points.json", "r") as f:
            points_dict = json.load(f)
        points = points_dict[name] / 1000
        result_str = f"& {psnr:.2f}, {fps:.2f}, {points:.2f} "
    # print(f"& {method}: {psnr_dict[name]:.2f}")
    # print(result_str)
    output_str += result_str
print(name, output_str)
