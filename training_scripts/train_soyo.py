import copy
reproduced_datasets_v1 = "seq-cifar10,seq-cropdisease,seq-mnist,seq-resisc45,seq-eurosat-rgb,seq-tinyimg,seq-cifar100,seq-chestx,seq-imagenet-r,seq-cub200"
reproduced_datasets_v2 = "seq-eurosat-rgb,seq-tinyimg,seq-imagenet-r,seq-cub200,seq-cifar100,seq-mnist,seq-resisc45,seq-chestx,seq-cifar10,seq-cropdisease"
reproduced_datasets_v3 = "seq-tinyimg,seq-resisc45,seq-cub200,seq-chestx,seq-imagenet-r,seq-eurosat-rgb,seq-mnist,seq-cifar10,seq-cropdisease,seq-cifar100"
# cddb_benchmarks = "seq-cddb-gaugan,seq-cddb-biggan,seq-cddb-wild,seq-cddb-whichfaceisreal,seq-cddb-san"
test_benchmarks = "seq-eurosat-rgb,seq-mnist,seq-cifar10"
list_datasets = [reproduced_datasets_v1, reproduced_datasets_v2, reproduced_datasets_v3]
METHODS = ["LEAR", "dualprompt", "codaprompt", "dap", "ranpac", "moe_adapters", "l2p"]
origin_params = {
    "list_datasets" : reproduced_datasets_v1,
    "dataset" : "seq-cifar10",
    "ncls_per_task" : 0,
    "model" : "soyo",
    "backbone" : "soyo",
    "lr" : 0.03,
    "batch_size" : 64,
    "n_epochs" : 10,
    "num_workers" : 8,
    "transform_type" : "weak",
    "n_frq" : int(768 * 768 / 16384),
    "n_experts" : 16384,
    "topk" : 3,
    "ce_thresh" : 1e-4,
}


list_params = []
for i, list_dataset in enumerate(list_datasets):
    params = copy.deepcopy(origin_params)
    params["list_datasets"] = list_dataset
    params["dataset"] = list_dataset.split(",")[0]
    list_params.append(params)
    
list_cmds = []
for params in list_params:
    cmd = "python -u main_domain.py"
    for key in params:
        param_cmd = f" --{key} {params[key]}"
        cmd += param_cmd
    list_cmds.append(cmd)

import os, subprocess
import time

for cmd in list_cmds:
    print(f"Đang bắt đầu: {cmd}")
    # shell=True cho phép chạy lệnh như một chuỗi CMD
    # check=True sẽ báo lỗi nếu lệnh train bị crash
    subprocess.run(cmd, shell=True, check=True) 
    print(f"Hoàn thành: {cmd}")