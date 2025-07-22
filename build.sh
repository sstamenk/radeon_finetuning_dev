#!/bin/bash

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

# Default values
DEFAULT_BASE_DOCKER="unsloth-dev:latest"
DEFAULT_GPU_ARCH=$(detect_gpu_arch)
IMAGE_NAME="unsloth-dev"
IMAGE_TAG="latest"
CONTAINER_NAME="unsloth-dev-container"
PWD=$(pwd)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--base-docker)
            BASE_DOCKER="$2"
            shift 2
            ;;
        -g|--gpu-arch)
            GPU_ARCH="$2"
            GPU_ARCH_OVERRIDE="true"
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
        # -v|--verbose)
        #     VERBOSE="true"
        #     shift
        #     ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Build Docker image with optional base image override"
            echo ""
            echo "Options:"
            echo "  -b, --base-docker    Base Docker image (default: $DEFAULT_BASE_DOCKER)"
            echo "  -g, --gpu-arch      ROCm architecture (default: auto-detected, currently: $DEFAULT_GPU_ARCH)"
            echo "  -n, --name          Image name (default: $IMAGE_NAME)"
            echo "  -t, --tag           Image tag (default: $IMAGE_TAG)"
            # echo "  -v, --verbose       Show GPU detection details"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Use default base image and ROCm arch"
            echo "  $0 -b nvidia/cuda:11.8-runtime-ubuntu20.04  # Override base image"
            echo "  $0 -r gfx1030                        # Override ROCm architecture"
            echo "  $0 -b rocm/pytorch:latest -g gfx1100 -n myapp -t v1.0  # Override multiple options"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
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

# Clone official Unsloth repository if it doesn't exist
if [ ! -d "unsloth_amd" ]; then
    echo "Cloning official Unsloth repository to unsloth_amd..."
    git clone https://github.com/unslothai/unsloth.git -b amd unsloth_amd
    echo "Cloning complete."
else
    echo "Official Unsloth repository already exists at unsloth_amd. Skipping clone."
fi

# Clone Unsloth fork if it doesn't exist
if [ ! -d "unsloth_rocm" ]; then
    echo "Cloning Unsloth fork to unsloth_rocm..."
    git clone https://github.com/billishyahao/unsloth.git -b billhe/rocm_enable unsloth_rocm
    echo "Cloning complete."
else
    echo "Unsloth fork already exists at unsloth_rocm. Skipping clone."
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
echo ""

docker build \
    --progress plain \
    --build-arg BASE_DOCKER="$BASE_DOCKER" \
    --build-arg ROCM_ARCH="$GPU_ARCH" \
    -t "$IMAGE_NAME:$IMAGE_TAG" \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Docker image built successfully!"
    echo "Image: $IMAGE_NAME:$IMAGE_TAG"
    echo "Base: $BASE_DOCKER"
    echo "ROCm Arch: $GPU_ARCH"
else
    echo ""
    echo "❌ Docker build failed!"
    exit 1
fi

docker run -it \
    --rm \
    --device /dev/dri \
    --device /dev/kfd \
    --network host \
    --ipc host \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    --shm-size 32G \
    --volume ${PWD}:/workspace \
    --name as1 \
    unsloth-dev:latest /bin/bash

# # Attach to the container if it is running
# if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
#     echo ""
#     echo "Attaching to the running container: ${CONTAINER_NAME}"
#     docker exec -it ${CONTAINER_NAME} /bin/bash
# else
#     echo ""
#     echo "Container ${CONTAINER_NAME} is not running."
#     echo "You can start it with: docker start ${CONTAINER_NAME}"
# fi