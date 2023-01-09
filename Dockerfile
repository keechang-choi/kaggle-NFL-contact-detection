FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

COPY ./linux-package-list.txt /tmp/
RUN apt-get update \
    && apt-get install -y sudo ffmpeg git zsh curl
# && apt-get upgrade -y \
# apt-get from text file fails??

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
# Install oh-my-zsh ("https://ohmyz.sh/")
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

ENV PATH="$PATH:~/.local/bin"

WORKDIR /workspace
COPY . .
