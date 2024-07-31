
#FROM pytorch/pytorch:latest 
# 2.3.0 might be even better as torchtext is no longer supported...
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
ENV PIP_ROOT_USER_ACTION=ignore

# Install system packages 
RUN apt-get update && apt-get install -yq wget vim git tmux

# Install pip requirements. Can also be done with devcontainer.json
COPY requirements.txt requirements.txt
RUN conda install -y numba && pip install -U --no-cache-dir -r requirements.txt

ARG user=mforgione
ARG group=mforgione
ARG uid=1006
ARG gid=1006
RUN groupadd -g ${gid} ${group}
RUN useradd -u ${uid} -g ${gid} -s /bin/bash -m ${user}
USER ${uid}:${gid}
