ARG BASE_DOCKER=rocm/vllm:latest
FROM ${BASE_DOCKER}

ARG WORKDIR=/workspace
WORKDIR ${WORKDIR}

ARG UNSLOTH_DIR=${WORKDIR}/unsloth
ARG ROCM_ARCH=gfx1100
ARG DEV_MODE=false
ENV MODELS_DIR=${WORKDIR}/models
ENV DATASETS_DIR=${WORKDIR}/datasets

# RUN git clone https://github.com/billishyahao/unsloth.git -b billhe/rocm_enable unsloth_fork
RUN if [ "$DEV_MODE" = "false" ]; then \
        git clone https://github.com/unslothai/unsloth.git -b amd $UNSLOTH_DIR; \
        # git clone https://github.com/billishyahao/unsloth.git -b billhe/rocm_enable unsloth_fork; \
        cp -r scripts ${WORKDIR}/scripts; \
        cp override_files/unsloth_setup.py ${UNSLOTH_DIR}/setup.py; \
        cp override_files/unsloth_req_rocm.txt ${UNSLOTH_DIR}/requirements/rocm.txt; \
        pip install -r ${UNSLOTH_DIR}/requirements/rocm.txt; \
        cd ${UNSLOTH_DIR} && python setup.py clean --all; \
        cd ${UNSLOTH_DIR} && python setup.py bdist_wheel; \
        pip install ${UNSLOTH_DIR}/dist/*.whl; \
    fi