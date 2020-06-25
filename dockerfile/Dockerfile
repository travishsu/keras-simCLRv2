FROM nvidia/cuda:10.1-devel-ubuntu18.04

# RUN export all_proxy=$HTTP_PROXY
# RUN echo "Acquire::http::Proxy \"$HTTP_PROXY\";" > /etc/apt/apt.conf

# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
ENV TENSORFLOW_VERSION=2.1.0
ENV CUDNN_VERSION=7.6.4.38-1+cuda10.1
ENV NCCL_VERSION=2.5.6-1+cuda10.1

# Python 2.7 or 3.6 is supported by Ubuntu Bionic out of the box
ARG python=3.6
ENV PYTHON_VERSION=${python}

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update
RUN apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        g++-4.8 \
        git \
        curl \
        vim \
        wget \
        ca-certificates \
        libcudnn7=${CUDNN_VERSION} \
        libcudnn7-dev=${CUDNN_VERSION} \
        libnccl2=${NCCL_VERSION} \
        libnccl-dev=${NCCL_VERSION} \
        libjpeg-dev \
        libpng-dev \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python3-pip \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev

# Install TensorRT. Requires that libcudnn7 is installed above.
RUN apt-get install -y --no-install-recommends libnvinfer6=6.0.1-1+cuda10.1 \
    libnvinfer-dev=6.0.1-1+cuda10.1 \
    libnvinfer-plugin6=6.0.1-1+cuda10.1

RUN if [[ "${PYTHON_VERSION}" == "3.6" ]]; then \
        apt-get install -y python${PYTHON_VERSION}-distutils; \
    fi

ARG TRUSTED_HOST_ARGS=''
RUN /usr/bin/pip3 install $TRUSTED_HOST_ARGS -U pip setuptools wheel

RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python
RUN ln -s /usr/local/bin/pip3.6 /usr/bin/pip

# Install TensorFlow, Keras, PyTorch and MXNet
RUN pip install $TRUSTED_HOST_ARGS  future typing
RUN pip install $TRUSTED_HOST_ARGS  numpy==1.18 \
        tensorflow-addons==0.9.1 \
        tensorflow-gpu==${TENSORFLOW_VERSION} \
        tensorflow-probability \ 
        keras \
        h5py \
        ipython

# Install Open MPI
# RUN mkdir /tmp/openmpi && \
#     cd /tmp/openmpi && \
#     wget --no-check-certificate https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
#     tar zxf openmpi-4.0.0.tar.gz && \
#     cd openmpi-4.0.0 && \
#     ./configure --enable-orterun-prefix-by-default && \
#     make -j $(nproc) all && \
#     make install && \
#     ldconfig && \
#     rm -rf /tmp/openmpi

# Install Horovod, temporarily using CUDA stubs
# RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
#     HOROVOD_GPU_OPERATION=NCCL \
#     HOROVOD_WITH_TENSORFLOW=1 \
#     HOROVOD_WITHOUT_PYTORCH=1 \
#     HOROVOD_WITHOUT_MXNET=1 \
#          pip install $TRUSTED_HOST_ARGS --no-cache-dir horovod && \
#     ldconfig

# Allow OpenSSH to talk to containers without asking for confirmation
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# Install jupyterlab
RUN /usr/bin/pip install $TRUSTED_HOST_ARGS jupyterlab
RUN wget https://raw.githubusercontent.com/tensorflow/tensorflow/r1.13/tensorflow/tools/docker/run_jupyter.sh --no-check-certificate
RUN wget https://raw.githubusercontent.com/tensorflow/tensorflow/r1.13/tensorflow/tools/docker/jupyter_notebook_config.py --no-check-certificate
RUN mkdir /root/.jupyter && mv jupyter_notebook_config.py /root/.jupyter/
RUN chmod +x /run_jupyter.sh && \
    sed -i -e 's/notebook/lab\ --NotebookApp.notebook_dir=\/sdi_mrcnn/g' /run_jupyter.sh && \
    printf '#!/bin/bash\njupyter nbconvert --ExecutePreprocessor.timeout=None --allow-errors --to html --execute $1' > /sbin/ipyrun && \
    chmod +x /sbin/ipyrun && \
    { printf 'alias smi="nvidia-smi"\nalias wsmi="watch -n 1 nvidia-smi"\n'; cat /etc/bash.bashrc; } >/etc/bash.bashrc.new && \
    mv /etc/bash.bashrc.new /etc/bash.bashrc && \
    mkdir -p /var/run/sshd && \
    cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# Install Tensorboard
RUN /usr/bin/pip install $TRUSTED_HOST_ARGS tensorboard
