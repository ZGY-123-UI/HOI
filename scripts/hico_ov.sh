#!/usr/bin/env bash

set -x  # Print each command for debugging

# --------- Environment Variables ---------
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=4

export DS_LOG_LEVEL=WARN
export ACCELERATE_LOG_LEVEL=WARN

# --------- Directory and Paths ---------
EXP_DIR="/media/qdu/2.0T/zgy/projects/SL-HOI/exps1/hico_ov"
DATA_DIR="/media/qdu/2.0T/zgy/projects/SL-HOI/data/hico_20160224_det/"
DINO_DIR="/media/qdu/2.0T/zgy/projects/SL-HOI/weights/dinov3"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "${MMPOSE_DIR}" ]; then
    if [ -d "${PROJECT_DIR}/third_party/mmpose/mmpose/apis" ]; then
        MMPOSE_DIR="${PROJECT_DIR}/third_party/mmpose"
    elif [ -d "${PROJECT_DIR}/mmpose/mmpose/apis" ]; then
        MMPOSE_DIR="${PROJECT_DIR}/mmpose"
    else
        MMPOSE_DIR="${PROJECT_DIR}/third_party/mmpose"
    fi
fi

if [ -z "${VITPOSE_DIR}" ]; then
    if [ -d "${PROJECT_DIR}/weights/Vitpose" ]; then
        VITPOSE_DIR="${PROJECT_DIR}/weights/Vitpose"
    else
        VITPOSE_DIR="${PROJECT_DIR}/weights/vitpose"
    fi
fi

DEFAULT_VITPOSE_CONFIG="${MMPOSE_DIR}/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py"
if [ ! -f "${DEFAULT_VITPOSE_CONFIG}" ] && [ -f "${MMPOSE_DIR}/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py" ]; then
    DEFAULT_VITPOSE_CONFIG="${MMPOSE_DIR}/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py"
fi
VITPOSE_CONFIG="${VITPOSE_CONFIG:-${DEFAULT_VITPOSE_CONFIG}}"

if [ -z "${VITPOSE_CHECKPOINT}" ]; then
    VITPOSE_CHECKPOINT="${VITPOSE_DIR}/vitpose-b.pth"
    if [ ! -f "${VITPOSE_CHECKPOINT}" ]; then
        FOUND_VITPOSE_CHECKPOINT=$(find "${VITPOSE_DIR}" -maxdepth 1 -type f \( -name "*ViTPose-base*.pth" -o -name "*vitpose*b*.pth" -o -name "*.pth" \) 2>/dev/null | head -n 1)
        if [ -n "${FOUND_VITPOSE_CHECKPOINT}" ]; then
            VITPOSE_CHECKPOINT="${FOUND_VITPOSE_CHECKPOINT}"
        fi
    fi
fi
export PYTHONPATH="${MMPOSE_DIR}:${PYTHONPATH}"

CONFIG_FILE="configs/hico.yaml"
DEFAULT_CONFIG="configs/base.yaml"

echo "Using MMPose dir: ${MMPOSE_DIR}"
echo "Using ViTPose config: ${VITPOSE_CONFIG}"
echo "Using ViTPose checkpoint: ${VITPOSE_CHECKPOINT}"

if [ ! -f "${VITPOSE_CONFIG}" ]; then
    echo "Error: ViTPose config not found: ${VITPOSE_CONFIG}"
    echo "Place the full MMPose repo under third_party/mmpose, not only a single config file."
    exit 1
fi

if [ ! -f "${VITPOSE_CHECKPOINT}" ]; then
    echo "Error: ViTPose checkpoint not found: ${VITPOSE_CHECKPOINT}"
    exit 1
fi

case "${VITPOSE_CHECKPOINT}" in
    *.pth|*.pt|*.ckpt) ;;
    *)
        echo "Error: MODEL.VITPOSE.CHECKPOINT must point to a weight file such as vitpose-b.pth, not a config .py file."
        echo "Current checkpoint path: ${VITPOSE_CHECKPOINT}"
        exit 1
        ;;
esac

python -c "import mmpose.apis, mmpose.utils" >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Error: Python cannot import mmpose.apis."
    echo "Expected a full MMPose package at: ${MMPOSE_DIR}"
    echo "Fix on the server with one of:"
    echo "  git clone https://github.com/open-mmlab/mmpose.git ${MMPOSE_DIR}"
    echo "  pip install -e ${MMPOSE_DIR}"
    echo "and make sure mmcv, mmengine, and mmpretrain are installed in the slhoi environment."
    exit 1
fi

# --------- Training Phase ---------
accelerate launch \
    --config_file "configs/accelerate_config.yaml" \
    --num_processes=1 \
    --main_process_port=12847 \
    train.py \
    -c ${CONFIG_FILE} \
    --default-config ${DEFAULT_CONFIG} \
    SOLVER.BATCH_SIZE=2 \
    RUNTIME.OUTPUT_DIR="${EXP_DIR}" \
    RUNTIME.NUM_WORKERS=0 \
    INPUT.PATH="${DATA_DIR}" \
    MODEL.DINO.DINOTXT_WEIGHTS="${DINO_DIR}/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth" \
    MODEL.DINO.BACKBONE_WEIGHTS="${DINO_DIR}/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" \
    MODEL.DINO.BPE_PATH_OR_URL="${DINO_DIR}/bpe_simple_vocab_16e6.txt.gz" \
    ZERO_SHOT.TYPE="rare_first" \
    ZERO_SHOT.DEL_UNSEEN="true" \
    ZERO_SHOT.CLASSIFIER.TRAIN="params/hico/classifier_rare_first.pt" \
    ZERO_SHOT.CLASSIFIER.EVAL="params/hico/classifier_eval.pt" \
    MODEL.RELATION_POSE_SCENE_ADAPTER.ENABLED="true" \
    MODEL.VITPOSE.ENABLED="true" \
    MODEL.VITPOSE.CONFIG="${VITPOSE_CONFIG}" \
    MODEL.VITPOSE.CHECKPOINT="${VITPOSE_CHECKPOINT}" \
    MODEL.VITPOSE.NUM_KEYPOINTS=17


# --------- Evaluation Phase ---------

# Find the saved yaml config file in EXP_DIR
SAVED_CONFIG=$(ls ${EXP_DIR}/*.yaml 2>/dev/null | head -n 1)

if [ -f "$SAVED_CONFIG" ]; then
    EVAL_CONFIG="$SAVED_CONFIG"
    echo "Found saved config: ${EVAL_CONFIG}"
else
    EVAL_CONFIG="${CONFIG_FILE}"
    echo "Saved config not found, using default: ${EVAL_CONFIG}"
fi

# Find all checkpoint directories and sort them numerically by epoch
CHECKPOINTS=$(ls -d ${EXP_DIR}/checkpoint-epoch-* 2>/dev/null | sort -t '-' -k 3 -n)

if [ -z "$CHECKPOINTS" ]; then
    echo "Error: No checkpoints found in ${EXP_DIR}!"
    exit 1
fi

# Loop through each checkpoint and run evaluation
for CKPT in $CHECKPOINTS; do
    EPOCH_NUM=$(basename "$CKPT" | cut -d'-' -f3)
    EVAL_OUTPUT_DIR="${CKPT}/eval"

    echo "--------------------------------------"
    echo "Evaluating Epoch: ${EPOCH_NUM} (Path: ${CKPT})"
    
    accelerate launch \
        --config_file "configs/accelerate_config.yaml" \
        --num_processes=8 \
        --main_process_port=12888 \
        train.py \
        -c ${EVAL_CONFIG} \
        --default-config ${DEFAULT_CONFIG} \
        RUNTIME.EVAL=true \
        RUNTIME.PRETRAINED=${CKPT} \
        RUNTIME.OUTPUT_DIR=${EVAL_OUTPUT_DIR} \
        RUNTIME.GLOBAL_OUTPUT_DIR=${EXP_DIR} \
        RUNTIME.NUM_WORKERS=4
        
    echo "Evaluation for Epoch ${EPOCH_NUM} done."
done
