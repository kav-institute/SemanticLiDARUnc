#!/bin/bash

# Set common parameters
LEARNING_RATE=0.001
NUM_EPOCHS=100
BATCH_SIZE=2
NUM_WORKERS=16
SCRIPT_PATH="/home/appuser/repos/baselines/CENet/train_semantic_THAB.py"

# Array of model types
MODEL_TYPES=(
    'HarDNet'
    'ResNet_34'
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
    for DATASET in "${DATASETS[@]}"
    do
        echo "Training with model: $MODEL_TYPE, Batch size: $BATCH_SIZE, Num workers: $NUM_WORKERS, Split:  $DATASET"
        python $SCRIPT_PATH --model_type $MODEL_TYPE --learning_rate $LEARNING_RATE --num_epochs $NUM_EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --test_id $DATASET 
    done
done
