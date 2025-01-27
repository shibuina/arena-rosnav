FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV RTI_NC_LICENSE_ACCEPTED=yes
ENV TZ=Etc/UTC
ENV PATH="/root/.local/bin:/root/.pyenv/bin:$PATH"
ENV PYENV_ROOT="/root/.pyenv"
ENV ARENA_WS_DIR="/arena4_ws"

ARG ARENA_ROS_VERSION=humble
ARG ARENA_BRANCH=humble

# Set the timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Add ROS 2 apt repository
RUN \
--mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
--mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt-get update && apt-get install -y curl gnupg2 lsb-release && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Update and install a comprehensive set of libraries and tools
RUN \
--mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
--mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt-get update && apt-get install -y \
        build-essential \
        cmake \
        sudo \
        git \
        wget \
        libssl-dev \
        zlib1g-dev \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        libsqlite3-dev \
        libreadline-dev \
        libffi-dev \
        libbz2-dev \
        liblzma-dev \
        python3 \
        python3-pip \
        python3-dev \
        python-is-python3 \
        software-properties-common

RUN if [ ! -e /usr/bin/python ]; then ln -s /usr/bin/python3 /usr/bin/python; fi
RUN python3 --version
RUN echo $PATH

# Download and run the install script
ADD install.sh install.sh
RUN \
--mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
--mount=target=/var/cache/apt,type=cache,sharing=locked \
--mount=type=cache,target=/root/.cache/pip \
    bash -c 'echo -e "\n\n\n\n\n\n\n" | bash install.sh'

CMD ["bash"]