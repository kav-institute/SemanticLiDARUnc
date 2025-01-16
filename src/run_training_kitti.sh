#!/bin/bash

# Set common parameters
LEARNING_RATE=0.001
NUM_EPOCHS=50
BATCH_SIZE=8
NUM_WORKERS=24
SCRIPT_PATH="/home/appuser/repos/train_semantic_KITTI.py"

# Specific parameters for certain models
SMALL_BATCH_SIZE=4
SMALL_NUM_WORKERS=24

# Array of model types
MODEL_TYPES=(
    'resnet18'
    'shufflenet_v2_x0_5'
    'resnet50'
    'shufflenet_v2_x1_5'
    #'resnet34'
    'shufflenet_v2_x1_0'
)

# Loop through each model type
for MODEL_TYPE in "${MODEL_TYPES[@]}"
do
    if [[ "$MODEL_TYPE" == "resnet50" || "$MODEL_TYPE" == "regnet_y_3_2gf"  || "$MODEL_TYPE" == "shufflenet_v2_x1_5" || "$MODEL_TYPE" == "shufflenet_v2_x1_0" ]]; then
        BATCH_SIZE=$SMALL_BATCH_SIZE
        NUM_WORKERS=$SMALL_NUM_WORKERS
    else
        BATCH_SIZE=8
        NUM_WORKERS=24
    fi

    # Loop through combinations of --attention and --normals flags
    for ATTENTION_FLAG in "--attention"
    do
        for NORMALS_FLAG in "--normals" #""
        do
            for MULTI_SCALE_FLAG in "--multi_scale_meta" #""
            do
                echo "Training with model: $MODEL_TYPE, Batch size: $BATCH_SIZE, Num workers: $NUM_WORKERS, Attention: $ATTENTION_FLAG, Normals: $NORMALS_FLAG, MultiScale: $MULTI_SCALE_FLAG"
                python $SCRIPT_PATH --model_type $MODEL_TYPE --learning_rate $LEARNING_RATE --num_epochs $NUM_EPOCHS --batch_size $BATCH_SIZE --flip --rotate --num_workers $NUM_WORKERS $ATTENTION_FLAG $NORMALS_FLAG $MULTI_SCALE_FLAG
            done
        done
    done
done
