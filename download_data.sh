#!/bin/bash

# Download datasets from Hugging Face using huggingface-cli
# This script downloads GSM8K and OpenMathReasoning-mini datasets if they don't already exist

set -e  # Exit on any error

# Parse command line arguments
CLEAN_MODE=false
for arg in "$@"; do
    case $arg in
        --clean|-c)
            CLEAN_MODE=true
            shift
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--clean|-c]"
            echo "  --clean, -c: Remove existing datasets and download them again"
            exit 1
            ;;
    esac
done

# Create datasets directory if it doesn't exist
DATASETS_DIR=$(dirname "$(realpath "$0")")/datasets
if [ ! -d "$DATASETS_DIR" ]; then
    mkdir -p "$DATASETS_DIR"
fi

# Function to check if dataset exists
check_dataset_exists() {
    local dataset_path="$1"
    if [ -d "$dataset_path" ] && [ "$(ls -A "$dataset_path")" ]; then
        return 0  # Dataset exists and is not empty
    else
        return 1  # Dataset doesn't exist or is empty
    fi
}

# Function to download dataset if it doesn't exist
download_dataset() {
    local repo_name="$1"
    local dataset_name="${repo_name##*/}"
    local dataset_path="$DATASETS_DIR/$dataset_name"
    
    echo "Checking for $dataset_name dataset at $dataset_path..."
    
    if [ "$CLEAN_MODE" = true ] && check_dataset_exists "$dataset_path"; then
        echo "Clean mode detected: Removing existing dataset $dataset_name..."
        rm -rf "$dataset_path"
        echo "Existing $dataset_name dataset removed."
    fi
    
    if check_dataset_exists "$dataset_path"; then
        echo "$dataset_name dataset already exists, skipping download."
    else
        echo "Downloading $dataset_name dataset..."
        huggingface-cli download "$repo_name" --repo-type dataset --local-dir "$dataset_path"
        echo "$dataset_name dataset downloaded successfully."
    fi
    
    # Set the path variable dynamically
    if [ "$dataset_name" = "gsm8k" ]; then
        GSM8K_PATH="$dataset_path"
    elif [ "$dataset_name" = "OpenMathReasoning-mini" ]; then
        OPENMATH_PATH="$dataset_path"
    elif [ "$dataset_name" = "FineTome-100k" ]; then
        FINETOME_PATH="$dataset_path"
    fi
}

# Download datasets
download_dataset "openai/gsm8k"
download_dataset "unsloth/OpenMathReasoning-mini"
download_dataset "mlabonne/FineTome-100k"

echo "Dataset downloads completed!"
echo "Datasets available at:"
echo "  - GSM8K: $GSM8K_PATH"
echo "  - OpenMathReasoning-mini: $OPENMATH_PATH"
echo "  - FineTome-100k: $FINETOME_PATH"
