#!/bin/bash

# Set common parameters
LEARNING_RATE=0.0001
NUM_EPOCHS=100
BATCH_SIZE=2    # 4
NUM_WORKERS=4  # 16
SCRIPT_PATH="/home/devuser/workspace/src/train_semantic_THAB_unc.py"

# Specific parameters for certain models
# SMALL_BATCH_SIZE=4
# SMALL_NUM_WORKERS=8

# Array of model types
MODEL_TYPES=(
    'efficientnet_v2_l'
    #'regnet_y_3_2gf'
    #'resnet50'
    #'shufflenet_v2_x1_5'
    #'efficientnet_v2_m'
    #'regnet_y_1_6gf'
    #'shufflenet_v2_x1_0'
    #'resnet34'
    #'efficientnet_v2_s'
    #'regnet_y_400mf'
    #'shufflenet_v2_x0_5'
    #'resnet18'
    #'regnet_y_800mf'
)


DATASETS=(
    #"0"
    #"1"
    #"2"
    #"3"
    #"4"
    #"5"
    "6"
    #"7"
    #"8"
    #"-1"
)


# Loop through each model type and run the training script
for MODEL_TYPE in "${MODEL_TYPES[@]}"
do
    # if [[ "$MODEL_TYPE" == "resnet50" || "$MODEL_TYPE" == "regnet_y_3_2gf"  || "$MODEL_TYPE" == "shufflenet_v2_x1_5" || "$MODEL_TYPE" == "shufflenet_v2_x1_0" ]]; then
    #     BATCH_SIZE=$SMALL_BATCH_SIZE
    #     NUM_WORKERS=$SMALL_NUM_WORKERS
    # else
    #     BATCH_SIZE=8
    #     NUM_WORKERS=16
    # fi
    
    # Loop through combinations of --attention and --normals flags
    for DATASET in "${DATASETS[@]}"
        do
        for PRETRAIN_FLAG in "--pretrained"
        do
            for ATTENTION_FLAG in "--attention" #""
            do
                for NORMALS_FLAG in "--normals" #""
                do
                    for MULTI_SCALE_FLAG in "--multi_scale_meta" #""
                    do
                        echo "Training with model: $MODEL_TYPE, Batch size: $BATCH_SIZE, Num workers: $NUM_WORKERS, Attention: $ATTENTION_FLAG, Normals: $NORMALS_FLAG, Pretrain: $PRETRAIN_FLAG, Split:  $DATASET"
                        python $SCRIPT_PATH --model_type $MODEL_TYPE --learning_rate $LEARNING_RATE --num_epochs $NUM_EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --test_id $DATASET --flip $ATTENTION_FLAG $NORMALS_FLAG $MULTI_SCALE_FLAG $PRETRAIN_FLAG
                    done
                done
            done
        done
    done
done