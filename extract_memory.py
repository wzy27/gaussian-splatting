import os, subprocess
import json

with open("data/need_names.json", "r") as f:
    name_dict = json.load(f)

# name = "suitcase008"
# dataset = "Omnizl"
# if not name[-1].isdigit():
#     dataset = "blender"


methods = ["3dgs", "gof", "2dgs", "ours"]
memory_lists = {}
for method in methods:
    memory_lists[method] = []
for name in name_dict:
    # print(name)

    dataset = "Omnizl"
    if not name[-1].isdigit():
        dataset = "blender"
    path_dicts = {
        "ours": f"/data/nglm005/zhengyu.wen/final/gaussian-splatting/output-ours-SH/{dataset}/{name}/point_cloud/iteration_30000/point_cloud.ply",
        # "ours": f"/data/nglm005/zhengyu.wen/final/gaussian-splatting/output-0.010-200/{dataset}/{name}/test/ours_30000/renders/r_{img}.png",
        "3dgs": f"/data/nglm005/zhengyu.wen/final/gaussian-splatting/output-base-SH/{dataset}/{name}/point_cloud/iteration_30000/point_cloud.ply",
        "gof": f"/data/nglm005/zhengyu.wen/gaussian-opacity-fields/output/{dataset}/{name}/point_cloud/iteration_30000/point_cloud.ply",
        "2dgs": f"/data/nglm005/zhengyu.wen/2d-gaussian-splatting/output-white/{dataset}/{name}/point_cloud/iteration_30000/point_cloud.ply",
    }
    if dataset == "blender":
        continue
        path_dicts["ours"] = (
            f"/data/nglm005/zhengyu.wen/final/gaussian-splatting/output-0.010-200/{dataset}/{name}/point_cloud/iteration_30000/point_cloud.ply"
        )

    for method in methods:
        source_path = path_dicts[method]
        # print(source_path)
        file_size = os.path.getsize(source_path)
        # print(method, f"{(file_size / 1024 / 1024):.2f}MB")
        memory_lists[method].append(file_size)

avg_dict = {}
for method in methods:
    avg_memory = sum(memory_lists[method]) / len(memory_lists[method])
    avg_memory_MB = avg_memory / 1024 / 1024
    print(f"{method}: {avg_memory_MB:.2f}MB")
    avg_dict[method] = avg_memory_MB

for method in methods:
    percentage = avg_dict[method] / avg_dict["ours"] - 1
    print(f"{method}: +{percentage*100:.2f}%")
# method = "neus2"
# command = f"scp {path_dicts[method]} {save_img_path}/{method}.png"
# print(command)
# subprocess.run(command, shell=True, executable="/bin/bash")
