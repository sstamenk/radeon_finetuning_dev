ARG BASE_DOCKER=rocm/vllm:latest
FROM ${BASE_DOCKER}

ARG WORKDIR=/workspace
WORKDIR ${WORKDIR}

ARG UNSLOTH_DIR=${WORKDIR}/unsloth_amd
ARG ROCM_ARCH=gfx1100
# Add your application-specific instructions here
# For example:
# COPY . /app
# WORKDIR /app
# RUN pip install -r requirements.txt
# CMD ["python", "app.py"]

# RUN git clone https://github.com/billishyahao/unsloth.git -b billhe/rocm_enable unsloth_rocm
# RUN git clone https://github.com/unslothai/unsloth.git -b amd unsloth_amd

# Copy setup files to both repositories
COPY unsloth_amd .
COPY unsloth_rocm .
COPY unsloth_setup.py unsloth_rocm/setup.py
COPY unsloth_setup.py unsloth_amd/setup.py
COPY unsloth_req_rocm.txt unsloth_rocm/requirements/rocm.txt
COPY unsloth_req_rocm.txt unsloth_amd/requirements/rocm.txt
COPY unsloth_llama8b.py ${WORKDIR}
COPY unsloth_qwen3_14b.py ${WORKDIR}


# RUN cd ${UNSLOTH_DIR} && python setup.py bdist_wheel
# RUN pip install ${UNSLOTH_DIR}/dist/*.whl