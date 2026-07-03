#!/usr/bin/env bash

set -x  # Print each command for debugging

# --------- Environment Variables ---------
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=2

export DS_LOG_LEVEL=WARN
export ACCELERATE_LOG_LEVEL=WARN

DATA_DIR="/media/qdu/2.0T/zgy/projects/SL-HOI/data/swig_hoi"
#CONFIG_FILE="configs/swig.yaml"
CONFIG_FILE="/media/qdu/2.0T/zgy/projects/SL-HOI/exps/swig/exp_config.yaml"
DEFAULT_CONFIG="configs/base.yaml"
#CHECKPOINT_PATH="pretrained/swig/pytorch_model.bin"
CHECKPOINT_PATH="/media/qdu/2.0T/zgy/projects/SL-HOI/exps/swig/checkpoint-epoch-59"
#EVAL_OUTPUT_DIR="pretrained/swig"
EVAL_OUTPUT_DIR="/media/qdu/2.0T/zgy/projects/SL-HOI/exps/swig/checkpoint-epoch-59/eval"

# --------- Evaluation ---------
accelerate launch \
    --config_file "configs/accelerate_config.yaml" \
    --num_processes=1 \
    --main_process_port=12888 \
    train.py \
    -c ${CONFIG_FILE} \
    --default-config ${DEFAULT_CONFIG} \
    SOLVER.BATCH_SIZE=2 \
    RUNTIME.EVAL=true \
    RUNTIME.PRETRAINED=${CHECKPOINT_PATH} \
    RUNTIME.OUTPUT_DIR=${EVAL_OUTPUT_DIR} \
    RUNTIME.NUM_WORKERS=0 \
    INPUT.PATH="${DATA_DIR}" \
    ZERO_SHOT.CLASSIFIER.EVAL="params/swig/classifier_swig_dict.pt"

echo "Evaluation for ${CHECKPOINT_PATH} finished. Check results in ${EVAL_OUTPUT_DIR}"
