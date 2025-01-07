
# Define base image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
#setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*



# setup environment
#ENV LANG C.UTF-8
#ENV LC_ALL C.UTF-8
RUN apt-get update && apt-get install -y \
    python3-opencv ca-certificates python3-dev git wget sudo ninja-build python3-pip build-essential cmake \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
ARG USER_NAME
ARG USER_ID
RUN useradd -m --no-log-init --system --uid ${USER_ID} ${USER_NAME} -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}

# The place for your repositories and data
RUN mkdir -p /home/${USER_NAME}/repos && mkdir /home/${USER_NAME}/data 

# Set workspace path
WORKDIR /home/${USER_NAME}/repos

# Copy entrypoint script and additional python packages
COPY entrypoint.sh /home/${USER_NAME}/entrypoint.sh
RUN sudo chmod +x /home/${USER_NAME}/entrypoint.sh
COPY requirements.txt /home/${USER_NAME}
RUN pip install --user torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --user -r /home/${USER_NAME}/requirements.txt


