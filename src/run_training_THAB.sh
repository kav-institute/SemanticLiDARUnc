#!/bin/bash

# Set common parameters
LEARNING_RATE=0.001
NUM_EPOCHS=30
BATCH_SIZE=8
NUM_WORKERS=16
SCRIPT_PATH="/home/appuser/repos/train_semantic_THAB.py"

# Specific parameters for certain models
SMALL_BATCH_SIZE=4
SMALL_NUM_WORKERS=8

# Array of model types
MODEL_TYPES=(
    #'resnet50'
    #'shufflenet_v2_x1_5'
    'resnet34'
    #'shufflenet_v2_x1_0'
    #'resnet18'
    #'shufflenet_v2_x0_5'
)

DATASETS=(
    "0"
    "1"
    "2"
    "3"
    "4"
    "5"
    "-1"
)


# Loop through each model type and run the training script
for MODEL_TYPE in "${MODEL_TYPES[@]}"
do
    if [[ "$MODEL_TYPE" == "resnet50" || "$MODEL_TYPE" == "regnet_y_3_2gf"  || "$MODEL_TYPE" == "shufflenet_v2_x1_5" || "$MODEL_TYPE" == "shufflenet_v2_x1_0" ]]; then
        BATCH_SIZE=$SMALL_BATCH_SIZE
        NUM_WORKERS=$SMALL_NUM_WORKERS
    else
        BATCH_SIZE=8
        NUM_WORKERS=16
    fi
    
    # Loop through combinations of --attention and --normals flags
    for DATASET in "${DATASETS[@]}"
        do
        for PRETRAIN_FLAG in "--pretrained" ""
        do
            for ATTENTION_FLAG in "--attention"
            do
                for NORMALS_FLAG in "--normals" ""
                do
                    for MULTI_SCALE_FLAG in "--multi_scale_meta" ""
                    do
                        echo "Training with model: $MODEL_TYPE, Batch size: $BATCH_SIZE, Num workers: $NUM_WORKERS, Attention: $ATTENTION_FLAG, Normals: $NORMALS_FLAG, Pretrain: $PRETRAIN_FLAG, Split:  $DATASET"
                        python $SCRIPT_PATH --model_type $MODEL_TYPE --learning_rate $LEARNING_RATE --num_epochs $NUM_EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --test_id $DATASET --flip $ATTENTION_FLAG $NORMALS_FLAG $MULTI_SCALE_FLAG $PRETRAIN_FLAG
                    done
                done
            done
        done
    done
done
