#!/bin/bash

# This script is the entry point for your submission.
# It handles both training and testing modes.

if [ "$1" = "test1" ]; then
    # Task 1 Inference Mode (Author Verification)
    # Usage: ./run_model.sh test1 <task1_test_file> <output_dir>
    # <task1_test_file> will be a JSON file containing author verification queries.
    # See sample_inputs/sample_task1.json for the format.
    
    TEST_FILE=$2
    OUTPUT_DIR=$3
    
    echo "Running Task 1 (Author Verification) inference on $TEST_FILE..."
    # TODO: Call your python inference script here
    # python3 src/inference_task1.py "$TEST_FILE" "$OUTPUT_DIR"
    
elif [ "$1" = "test2" ]; then
    # Task 2 Inference Mode (Author Clustering)
    # Usage: ./run_model.sh test2 <task2_test_file> <output_dir>
    # <task2_test_file> will be a JSON file containing chunks to cluster.
    # See sample_inputs/sample_task2.json for the format.
    # NOTE: You are allowed to finetune/adapt your embeddings on this test data
    #       since Task 2 is an unsupervised clustering task (transductive learning).
    
    TEST_FILE=$2
    OUTPUT_DIR=$3
    
    echo "Running Task 2 (Author Clustering) inference on $TEST_FILE..."
    # TODO: Call your python inference script here
    # python3 src/inference_task2.py "$TEST_FILE" "$OUTPUT_DIR"
    
else
    # Training Mode
    # Usage: ./run_model.sh <train_dir>
    # <train_dir> will be a directory containing author text files (see data/train_data/).
    # You are responsible for splitting this data into train/validation sets as needed.
    
    TRAIN_DIR=$1
    
    echo "Training model on data in $TRAIN_DIR..."
    # TODO: Call your python training script here
    # python3 src/train.py "$TRAIN_DIR"
fi
