#!/bin/bash

# Kích hoạt environment
# source /home/tahuuloc/miniconda3/envs/dcpt/bin/activate

# Chuyển đến workspace
cd "$(dirname "${BASH_SOURCE[0]}")"

# Lựa chọn config
CONFIG="${1:-train_NAFNet_AIO_5d_custom}"

# Chạy trnoaining
export BASICSR_JIT=True
nohup /home/tahuuloc/miniconda3/envs/dcpt/bin/python -m torch.distributed.run --master-port 12345 --nproc_per_node 1 \
    basicsr/all_in_one_train.py -opt options/all_in_one/train/${CONFIG}.yml --launcher pytorch > train.log 2>&1 &
