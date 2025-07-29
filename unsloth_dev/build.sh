#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Build Unsloth Docker image with optional base image override"
    echo ""
    echo "Options:"
    echo "  -b, --base-docker    Base Docker image (default: $DEFAULT_BASE_DOCKER)"
    echo "  -g, --gpu-arch       ROCm architecture (default: auto-detected, currently: $DEFAULT_GPU_ARCH)"
    echo "  -n, --name           Image name (default: $IMAGE_NAME)"
    echo "  -t, --tag            Image tag (default: $IMAGE_TAG)"
    echo "  -d, --dev            Development mode: clone repos locally and mount them (default: $DEV_MODE)"
    echo "  -m, --models-dir     Directory containing models to be mounted (default: $MODELS_DIR_PATH)"
    echo "  -D, --datasets-dir   Directory containing datasets to be mounted (default: $DATASETS_DIR_PATH)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use default base image and ROCm arch"
    echo "  $0 -b nvidia/cuda:11.8-runtime-ubuntu20.04  # Override base image"
    echo "  $0 -r gfx1030                        # Override ROCm architecture"
    echo "  $0 -b rocm/pytorch:latest -g gfx1100 -n myapp -t v1.0  # Override multiple options"
    echo "  $0 -d                                 # Development mode with local repo mounting"
}

# Function to detect GPU architecture using rocm-smi
detect_gpu_arch() {
    local arch=""
    
    # Check if rocm-smi is available
    if ! command -v rocm-smi >/dev/null 2>&1; then
        echo "gfx1100"  # Default fallback
        return
    fi
    
    # Use rocm-smi --showhw to get hardware info
    local rocm_output=$(rocm-smi --showhw 2>/dev/null)
    
    if [ $? -eq 0 ] && [ -n "$rocm_output" ]; then
        # Extract gfx architecture from the output
        # Look for patterns like "gfx1100", "gfx90a", etc.
        arch=$(echo "$rocm_output" | grep -oE "gfx[0-9a-f]+" | head -1)
    fi
    
    # If extraction failed, try alternative patterns
    if [ -z "$arch" ]; then
        # Sometimes the output format might be different, try other common patterns
        arch=$(echo "$rocm_output" | grep -i "device\|gpu" | grep -oE "gfx[0-9a-f]+" | head -1)
    fi
    
    # Default fallback if nothing found
    if [ -z "$arch" ]; then
        arch="gfx1100"
    fi
    
    echo "$arch"
}

# Function to clone repositories in development mode
clone_repos_dev() {
    echo "=== Development Mode: Cloning repositories locally ==="
    
    # Clone official Unsloth repository if it doesn't exist
    if [ ! -d "${UNSLOTH_DIR_PATH}" ]; then
        echo "Cloning into $UNSLOTH_REPO_URL..."
        git clone $UNSLOTH_REPO_URL -b $UNSLOTH_REPO_BRANCH ${UNSLOTH_DIR_PATH}
        echo "Unsloth repository cloned."
    else
        echo "Unsloth repository already exists. Skipping clone."
    fi

    echo "=== Local repositories ready ==="
    echo ""
}

# Default values
UNSLOTH_REPO_URL="https://github.com/unslothai/unsloth.git"
UNSLOTH_REPO_BRANCH="amd"
UNSLOTH_DIR="unsloth"

DEFAULT_BASE_DOCKER="registry-sc-harbor.amd.com/framework/compute-rocm-dkms-no-npi-hipclang:16449_ubuntu24.04_py3.12_vllm_rocm-7.0_97c32fc_pytorch_release-2.7_35daec93"
# DEFAULT_BASE_DOCKER="rocm/vllm:latest"
DEFAULT_GPU_ARCH=$(detect_gpu_arch)
IMAGE_NAME="unsloth-dev"
IMAGE_TAG="latest"
CONTAINER_NAME="unsloth-dev"
BASE_DIR=$(dirname "$(realpath "$0")")
UNSLOTH_DIR_PATH="${BASE_DIR}/${UNSLOTH_DIR}"
DEV_MODE=true
MODELS_DIR_PATH="/home/master/develop/models"
DATASETS_DIR_PATH="$BASE_DIR/datasets"

# TODO: Create a README.md file with instructions on how to use this script and the Docker image

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--base-docker)
            BASE_DOCKER="$2"
            shift 2
            ;;
        -d|--dev)
            DEV_MODE=true
            shift
            ;;
        -g|--gpu-arch)
            GPU_ARCH="$2"
            GPU_ARCH_OVERRIDE="true"
            shift 2
            ;;
        -m|--models-dir)
            MODELS_DIR_PATH="$2"
            shift 2
            ;;
        -n|--name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -D|--datasets-dir)
            DATASETS_DIR_PATH="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
        esac
    done

# Set BASE_DOCKER to default if not provided
if [ -z "$BASE_DOCKER" ]; then
    BASE_DOCKER="$DEFAULT_BASE_DOCKER"
fi

# Set ROCM_ARCH to default if not provided
if [ -z "$GPU_ARCH" ]; then
    GPU_ARCH="$DEFAULT_GPU_ARCH"
fi

# Clone repositories if in development mode and override the required files
if [ "$DEV_MODE" = true ]; then
    clone_repos_dev
fi

# Check if MODELS_DIR_PATH is set and exists
if [ -z "$MODELS_DIR_PATH" ] || [ ! -d "$MODELS_DIR_PATH" ] ||
   [ -z "$(ls -A "$MODELS_DIR_PATH" 2>/dev/null)" ]; then
    echo "Error: MODELS_DIR_PATH is not set or does not exist or directory is empty."
    echo "MODELS_DIR_PATH=$MODELS_DIR_PATH"
    echo "Please set the MODELS_DIR_PATH variable to a valid directory containing models."
    exit 1
fi

# Check if DATASETS_DIR_PATH is set and exists
if [ -z "$DATASETS_DIR_PATH" ] || [ ! -d "$DATASETS_DIR_PATH" ] ||
   [ -z "$(ls -A "$DATASETS_DIR_PATH" 2>/dev/null)" ]; then
    echo "Error: DATASETS_DIR_PATH is not set or does not exist or the directory is empty."
    echo "DATASETS_DIR_PATH=$DATASETS_DIR_PATH"
    echo "Please set the DATASETS_DIR_PATH variable to a valid directory containing datasets."
    echo "Run the download_datasets.sh script to download the datasets."
    exit 1
fi

# Build the Docker image
echo "Building Docker image..."
echo "Base Docker image: $BASE_DOCKER"
echo "ROCm architecture: $GPU_ARCH"
if [ "$GPU_ARCH" = "$(detect_gpu_arch)" ] && [ -z "${GPU_ARCH_OVERRIDE:-}" ]; then
    echo "  (Auto-detected GPU architecture)"
else
    echo "  (Using specified GPU architecture)"
fi
echo "Image name: $IMAGE_NAME"
echo "Image tag: $IMAGE_TAG"
echo "Development mode: $DEV_MODE"
echo "Models directory: $MODELS_DIR_PATH"
echo "Datasets directory: $DATASETS_DIR_PATH"
echo ""

# Prepare Docker build arguments
BUILD_ARGS="--pull \
            --progress plain \
            --build-arg BASE_DOCKER=\"$BASE_DOCKER\" \
            --build-arg ROCM_ARCH=\"$GPU_ARCH\" \
            --build-arg DEV_MODE=\"$DEV_MODE\""

# Build the Docker image
eval "docker build --no-cache $BUILD_ARGS -t \"$IMAGE_NAME:$IMAGE_TAG\" $BASE_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Docker image built successfully!"
    echo "Image: $IMAGE_NAME:$IMAGE_TAG"
    echo "Base: $BASE_DOCKER"
    echo "ROCm Arch: $GPU_ARCH"
    echo "Dev Mode: $DEV_MODE"
    echo "Models Directory: $MODELS_DIR_PATH"
    echo "Datasets Directory: $DATASETS_DIR_PATH"
else
    echo ""
    echo "❌ Docker build failed!"
    exit 1
fi

# Prepare volume mounts for Docker run
DOCKER_WORKSPACE="/workspace"
UNSLOTH_DOCKER_DIR_PATH="${DOCKER_WORKSPACE}/${UNSLOTH_DIR}"
VOLUME_MOUNTS="--volume ${MODELS_DIR_PATH}:${DOCKER_WORKSPACE}/models \
               --volume ${DATASETS_DIR_PATH}:${DOCKER_WORKSPACE}/datasets"

# Add repository mounts if in development mode
if [ "$DEV_MODE" = true ]; then
    if [ -d "${UNSLOTH_DIR_PATH}" ]; then
        VOLUME_MOUNTS="$VOLUME_MOUNTS \
                       --volume ${BASE_DIR}/${UNSLOTH_DIR}:$UNSLOTH_DOCKER_DIR_PATH"
    fi
    if [ -d "${BASE_DIR}/scripts" ]; then
        VOLUME_MOUNTS="$VOLUME_MOUNTS \
                       --volume ${BASE_DIR}/scripts:$DOCKER_WORKSPACE/scripts"
    fi
    if [ -d "${BASE_DIR}/override_files" ]; then
        VOLUME_MOUNTS="$VOLUME_MOUNTS \
                       --volume ${BASE_DIR}/override_files/unsloth_setup.py:$UNSLOTH_DOCKER_DIR_PATH/setup.py"
        VOLUME_MOUNTS="$VOLUME_MOUNTS \
                       --volume ${BASE_DIR}/override_files/unsloth_req_rocm.txt:$UNSLOTH_DOCKER_DIR_PATH/requirements/rocm.txt"
    fi
    echo "Development mode: Mounting local files into container"
fi
echo "Volume mounts: $VOLUME_MOUNTS"
# if container with the same name exists, remove it
if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo "Removing existing container with name ${CONTAINER_NAME}..."
    docker rm -f ${CONTAINER_NAME}
fi

# Run the Docker container in detached mode
eval "docker run -d \
    --device /dev/dri \
    --device /dev/kfd \
    --network host \
    --ipc host \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    --shm-size 32G \
    $VOLUME_MOUNTS \
    --name ${CONTAINER_NAME} \
    $IMAGE_NAME:$IMAGE_TAG tail -f /dev/null"



# Attach to the container if it is running
if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
    echo ""
    echo "✅ Container ${CONTAINER_NAME} started successfully!"
    echo "Attaching to the running container: ${CONTAINER_NAME}"
    
    if [ "$DEV_MODE" = true ]; then
        BUILD_CMD=" \
            pip install -r ${UNSLOTH_DOCKER_DIR_PATH}/requirements/rocm.txt; \
            cd ${UNSLOTH_DOCKER_DIR_PATH} && python setup.py clean --all; \
            cd ${UNSLOTH_DOCKER_DIR_PATH} && python setup.py bdist_wheel; \
            pip install ${UNSLOTH_DOCKER_DIR_PATH}/dist/*.whl; \
        "

        docker exec -it ${CONTAINER_NAME} /bin/bash -c "$BUILD_CMD"
    fi
    docker exec -it ${CONTAINER_NAME} /bin/bash
else
    echo ""
    echo "Container ${CONTAINER_NAME} is not running."
    echo "❌ Failed to start the container!"
    exit 1
fi
