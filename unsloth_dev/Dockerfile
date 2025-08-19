ARG BASE_DOCKER=rocm/vllm:latest
FROM ${BASE_DOCKER}

ARG WORKDIR=/workspace
WORKDIR ${WORKDIR}

ARG UNSLOTH_DIR=${WORKDIR}/unsloth
ARG ROCM_ARCH=gfx1100
ARG DEV_MODE=false
ENV MODELS_DIR=${WORKDIR}/models
ENV DATASETS_DIR=${WORKDIR}/datasets

# Copy local files
COPY scripts ${WORKDIR}/tmp/scripts
COPY override_files ${WORKDIR}/tmp/override_files

RUN apt update -y && \ 
    apt install -y libcurl4-openssl-dev 

RUN git clone https://github.com/ggml-org/llama.cpp ${WORKDIR}/llama.cpp
RUN cd ${WORKDIR}/llama.cpp && \
    HIPCXX="$(hipconfig -l)/clang"; \
    HIP_PATH="$(hipconfig -R)"; \
    cmake -S "." \
        -B "build" \
        -DGGML_HIP=ON \
        -DAMDGPU_TARGETS="$ROCM_ARCH" \
        -DCMAKE_BUILD_TYPE=Release && \
    cmake --build "build" --clean-first --target llama-quantize --target llama-export-lora --target llama-cli --config Release -- -j"$(nproc)" && \
    cp ${WORKDIR}/llama.cpp/build/bin/llama-* ${WORKDIR}/llama.cpp

# RUN git clone https://github.com/billishyahao/unsloth.git -b billhe/rocm_enable unsloth_fork
RUN if [ "$DEV_MODE" = "false" ]; then \
        # git clone https://github.com/unslothai/unsloth.git -b amd $UNSLOTH_DIR; \
        git clone https://github.com/billishyahao/unsloth.git -b billhe/rocm_enable $UNSLOTH_DIR; \
        mv ${WORKDIR}/tmp/scripts ${WORKDIR}/scripts; \
        mv ${WORKDIR}/tmp/override_files/unsloth_setup.py ${UNSLOTH_DIR}/setup.py; \
        mv ${WORKDIR}/tmp/override_files/unsloth_req_rocm.txt ${UNSLOTH_DIR}/requirements/rocm.txt; \
        rm -rf ${WORKDIR}/tmp; \
        pip install -r ${UNSLOTH_DIR}/requirements/rocm.txt; \
        cd ${UNSLOTH_DIR} && python setup.py clean --all; \
        cd ${UNSLOTH_DIR} && python setup.py bdist_wheel; \
        pip install ${UNSLOTH_DIR}/dist/*.whl; \
    fi
