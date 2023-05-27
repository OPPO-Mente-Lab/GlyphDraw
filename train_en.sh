#!/bin/bash
export CPATH=/usr/local/cuda/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
ROOT_DIR=/public_data/ma/code/stablediffusion-controlnet/results_mul

MODEL_NAME=stablediffusion_en
MODEL_ROOT_DIR=$ROOT_DIR/${MODEL_NAME}
if [ ! -d ${MODEL_ROOT_DIR} ];then
  mkdir ${MODEL_ROOT_DIR}
fi

MICRO_BATCH_SIZE=22
GPUS_PER_NODE=$2
NNODES=1


CONFIG_JSON="$MODEL_ROOT_DIR/${MODEL_NAME}.ds_config.json"
ZERO_STAGE=1
cat <<EOT > $CONFIG_JSON
{
    "zero_optimization": {
        "stage": ${ZERO_STAGE}
    },
    "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE
}
EOT
export PL_DEEPSPEED_CONFIG_PATH=$CONFIG_JSON


DATA_ARGS="\
        --webdataset_base_urls \
        /public_data/zmj/data_en_no_ocr/aesthetics_tar_5_packed/{00000..00060}.tar \
        /public_data/zmj/data_font_en_second/aesthetics_tar_5_cap/{00000..00192}.tar \
        --num_workers 2 \
        --batch_size $MICRO_BATCH_SIZE \
        --shard_width 5 \
        --hr_size 512 \
        --train_split 1.0 \
        --val_split 0.0 \
        --test_split 0.0 \
        --resample_train \
        "


MODEL_ARGS="\
        --model_path /public_data/ma/stable_models/model_base \
        --clip_path /public_data/ma/models/ViT-H-14.pt \
        --learning_rate 3e-5 \
        --weight_decay 1e-1 \
        --warmup_steps 100 \
        "

MODEL_CHECKPOINT_ARGS="\
        --save_last \
        --save_ckpt_path ${MODEL_ROOT_DIR}/ckpt \
        --load_ckpt_path ${MODEL_ROOT_DIR}/ckpt/last.ckpt \
        "

TRAINER_ARGS="\
        --max_epoch 10 \
        --accelerator gpu \
        --devices $GPUS_PER_NODE \
        --num_nodes $NNODES \
        --strategy deepspeed_stage_${ZERO_STAGE} \
        --log_every_n_steps 100 \
        --precision 16 \
        --default_root_dir ${MODEL_ROOT_DIR} \
        --replace_sampler_ddp False \
        --num_sanity_val_steps 0 \
        --limit_val_batches 0 \
        "
# num_sanity_val_steps， limit_val_batches 通过这俩参数把validation关了

export options=" \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "

export CC=gcc
export CXX=g++
python -m torch.distributed.run \
    --nnodes $NNODES \
    --master_addr 10.25.193.84 \
    --master_port 29500 \
    --node_rank $1 \
    --nproc_per_node $GPUS_PER_NODE \
    train_en.py $options
