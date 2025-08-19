# Unsloth Dev Tools
A collection of tools to simplify building and developing Unsloth and other fine-tuning libraries. Includes utilities for downloading required datasets, running a Docker container with all prerequisites pre-installed, and a suite of testing scripts for Unsloth.
## Directory Layout
```
unsloth_dev/
  build.sh                # Build + run ROCm dev container
  Dockerfile              # Image recipe (used by build.sh)
  datasets/
    download_data.sh      # Hugging Face datasets downloader
  override_files/         # Local overrides (setup.py, rocm.txt) mounted in dev mode
  scripts/                # Example fine‑tuning / training scripts
  unsloth/                # Cloned upstream repo (branch: amd) in dev mode
```

## Prerequisites
- ROCm installed
- Python packages from `requirements.txt`
- Local non-empty models dir (e.g. `/home/master/develop/models/...`).

## Dataset Downloader (`datasets/download_data.sh`)
Downloads if missing:
- `openai/gsm8k`
- `unsloth/OpenMathReasoning-mini`
- `mlabonne/FineTome-100k`

Usage:
```bash
cd datasets
./download_data.sh
./download_data.sh --clean   # Forces re-download
```

## Docker Build Script (`build.sh`)
Purpose: Build ROCm dev image and start a privileged container with live-mounted source, models, and datasets.

Key behavior:
- Auto GPU arch via `rocm-smi --showhw` (fallback: `gfx1100`).
- Dev mode clones Unsloth (branch `amd`) if absent and mounts local directories as live volumes into the container, enabling real-time file editing on the host that immediately reflects inside the container without rebuilds.
- When dev mode is disabled, files are copied into the container during build instead of mounted, requiring container recreation for changes.
- After start, rebuilds and installs wheel inside container.

### Options
```bash
./build.sh [options]
  -b, --base-docker  <image>
  -g, --gpu-arch     <gfx arch>
  -n, --name         <image name>
  -t, --tag          <image tag>
  -d, --dev          (dev mode; currently default true)
  -m, --models-dir   <host models dir> (must exist & non-empty)
  -D, --datasets-dir <host datasets dir> (default: ./datasets)
  -h, --help
```

### Examples
```bash
./build.sh -m /home/master/develop/models
./build.sh -m /home/master/develop/models -g gfx1100 -t dev1
./build.sh -m /home/master/develop/models -b rocm/vllm:latest
```

### Mounted Paths (inside `/workspace`)
- Models → `/workspace/models`
- Datasets → `/workspace/datasets`
- Dev mode extras:
  - `unsloth/` source → `/workspace/unsloth`
  - `scripts/` → `/workspace/scripts`
  - Override `override_files/unsloth_setup.py` → `/workspace/unsloth/setup.py`
  - Override `override_files/unsloth_req_rocm.txt` → `/workspace/unsloth/requirements/rocm.txt`

### Auto Dev Build Steps
1. `pip install -r requirements/rocm.txt`
2. `python setup.py clean --all`
3. `python setup.py bdist_wheel`
4. `pip install dist/*.whl`

Manual rebuild after edits:
```bash
cd /workspace/unsloth
python setup.py bdist_wheel && pip install dist/*.whl --force-reinstall
```

### Typical Workflow
1. Install Python dependencies: `pip install -r requirements.txt`
2. `cd datasets && ./download_data.sh`
3. Download models you want to test with `hf download`.
4. `./build.sh -m <MODELS_DIR>`
5. Inside container: `python /workspace/scripts/unsloth_llama8b.py`
6. Edit locally → rebuild wheel if needed.
