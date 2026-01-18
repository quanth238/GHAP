#!/bin/bash

# Usage: bash run_from_pointcloud.sh <scene_name> <source_path> [<ckp_path>] [<sampling_ratio>] [--extra_args ...]

scene_name=$1
source_path=$2
ckp_path=$3
shift 3

sampling_ratio=0.2
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
  sampling_ratio=$1
  shift
fi

compaction_method=""
target_num=""
extra_args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --compaction_method)
      compaction_method=$2
      extra_args+=("$1" "$2")
      shift 2
      ;;
    --target_num_gaussians)
      target_num=$2
      extra_args+=("$1" "$2")
      shift 2
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

# parameter checking
if [ -z "$scene_name" ] || [ -z "$source_path" ]; then
  echo "Usage: bash run_from_pointcloud.sh <scene_name> <source_path> [<ckp_path>] [<sampling_ratio>] [--extra_args ...]"
  exit 1
fi

# build output path
if [ -n "$target_num" ]; then
  directory1=./experiments/${scene_name}_k${target_num}_from_pointcloud
else
  directory1=./experiments/${scene_name}_${sampling_ratio}_from_pointcloud
fi

# optional checkpoint flag
if [ -n "$ckp_path" ]; then
  ckpt_flag="--start_checkpoint $ckp_path"
else
  ckpt_flag=""
fi

# training
CUDA_VISIBLE_DEVICES=3 python train_and_prune.py \
    -s "$source_path" \
    -m "$directory1" \
    --sampling_ratio $sampling_ratio \
    --eval \
    --compact \
    --disable_viewer \
    $ckpt_flag \
    --iterations 30000 \
    --test_iterations 15001 15002 30000 \
    --save_iterations 30000 \
    --checkpoint_iterations 15000 \
    --sampling_iter 15001 \
    "${extra_args[@]}"

# render & evaluate
CUDA_VISIBLE_DEVICES=3 python render.py -m "$directory1"
CUDA_VISIBLE_DEVICES=3 python metrics.py -m "$directory1"