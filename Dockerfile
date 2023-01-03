# docker pull pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime


FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN apt-get update
COPY ./linux-package-list.txt /tmp/
RUN apt-get install -y $(cat /tmp/linux-package-list.txt)

# https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
ARG USERNAME=docker_user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

COPY requirements.txt /tmp/
RUN cd /tmp/ && pip install -r requirements.txt

ENV PATH="$PATH:~/.local/bin"

WORKDIR /workspace
COPY . .