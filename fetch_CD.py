import json

with open("data/result_files/all_CD-as_mesh.json", "r") as f:
    CD_dict = json.load(f)
methods = ["voxurf", "neus2", "sugar", "2dgs", "gof", "ours"]  #

metric = "forward"
name = "ficus"

for method in methods:
    print(f"{method}: {CD_dict[name][method][metric]*10000:.2f}")
