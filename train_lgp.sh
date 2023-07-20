#!/bin/bash
#SBATCH --job-name=finetune_oppo # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=8 # number of tasks to run per node
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:8 # number of gpus per node
#SBATCH -o %x-%j.log # output and error log file names (%x for job id)
#SBATCH -x dgx050
export CPATH=/usr/local/cuda/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
# pwd=Fengshenbang-LM/fengshen/examples/pretrain_erlangshen
ROOT_DIR=/public_data/ma/code/stablediffusion-font/result  
# export TORCH_EXTENSIONS_DIR=/public_data/wrc/torch_extensions

MODEL_NAME=stablediffusion_lgp_zh
MODEL_ROOT_DIR=$ROOT_DIR/${MODEL_NAME}
if [ ! -d ${MODEL_ROOT_DIR} ];then
  mkdir ${MODEL_ROOT_DIR}
fi

MICRO_BATCH_SIZE=50
GPUS_PER_NODE=$2
NNODES=1

CONFIG_JSON="$MODEL_ROOT_DIR/${MODEL_NAME}.ds_config.json"
ZERO_STAGE=1
# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
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
        /data_share/zhaomingjun/data_cleaning/character_tar_simple_with_bg_exampler_randpos_multifont/{00000..00010}.tar \
        /public_data/zmj/data_font_filtered_2/laion_zh_webdataset/{00000..00013}.tar \
        /public_data/zmj/data_font_filtered_2/wukong_webdataset/{00000..00026}.tar \
        /public_data/zmj/data_font_filtered_2/querydata_total_dedup_len_72/{00000..00014}.tar \
        /public_data/zmj/data_font_filtered_2/query_data_zhutici_total_len_72_224+/{00000..00015}.tar \
        /public_data/zmj/data_font_filtered_2/zero23M_len_72/{00000..00010}.tar \
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
        --clip_path /public_data/text2img_code/models_stable_diffusion_c/text_encoder_400M \
        --learning_rate 1e-4 \
        --warmup_steps 100 \
        "

MODEL_CHECKPOINT_ARGS="\
        --save_last \
        --save_ckpt_path ${MODEL_ROOT_DIR}/ckpt \
        --load_ckpt_path ${MODEL_ROOT_DIR}/ckpt/last.ckpt \
        "

TRAINER_ARGS="\
        --max_epoch 30 \
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
# echo $options
python print_args.py --model_args="$options"
# python finetune.py $options

export CC=gcc
export CXX=g++
# python finetune_ori.py $options
python -m torch.distributed.run \
    --nnodes $NNODES \
    --master_addr 10.25.193.68 \
    --master_port 29502 \
    --node_rank $1 \
    --nproc_per_node $GPUS_PER_NODE \
    train_lgp.py $options
