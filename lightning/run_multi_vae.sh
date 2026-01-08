#!/bin/bash

# MultiVAE Training and Inference Script
# Usage: ./run_multi_vae.sh [train|predict|both] [config_file]

set -e  # Exit on error

MODE=${1:-both}  # Default: both
CFG_FILE=${2:-multi_vae_v2}

echo "========================================="
echo "MultiVAE Runner"
echo "========================================="
echo "Mode: $MODE"
echo "Config: $CFG_FILE"
echo ""

case $MODE in
    train)
        echo "Starting training..."
        python train_multi_vae.py -cn $CFG_FILE
        ;;

    predict)
        echo "Starting inference..."
        if [ -f "predict_multi_vae.py" ]; then
            python predict_multi_vae.py -cn $CFG_FILE
        else
            echo "Warning: predict_multi_vae.py not found"
            echo "Skipping prediction step"
        fi
        ;;

    both)
        echo "Starting training..."
        python train_multi_vae.py -cn $CFG_FILE

        echo ""
        echo "Training completed. Starting inference..."
        if [ -f "predict_multi_vae.py" ]; then
            python predict_multi_vae.py -cn $CFG_FILE
        else
            echo "Warning: predict_multi_vae.py not found"
            echo "Skipping prediction step"
        fi
        ;;

    *)
        echo "Invalid mode: $MODE"
        echo ""
        echo "Usage: ./run_multi_vae.sh [mode] [config_file]"
        echo ""
        echo "Modes:"
        echo "  train         - Train only"
        echo "  predict       - Predict only"
        echo "  both          - Train + Predict (default)"
        echo ""
        echo "Examples:"
        echo "  ./run_multi_vae.sh train multi_vae_v2"
        echo "  ./run_multi_vae.sh both multi_vae_v2"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "Done!"
echo "========================================="
