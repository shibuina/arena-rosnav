#!/bin/bash -i
set -e

export ARENA_ROSNAV_REPO=${ARENA_ROSNAV_REPO:-voshch/arena-rosnav}
export ARENA_BRANCH=${ARENA_BRANCH:-humble}
export ARENA_ROS_VERSION=${ARENA_ROS_VERSION:-humble}

# == read inputs ==
echo 'Configuring arena-rosnav...'

ARENA_WS_DIR=${ARENA_WS_DIR:-~/arena4_ws}
read -p "arena-rosnav workspace directory [${ARENA_WS_DIR}] " INPUT
export ARENA_WS_DIR=$(realpath "$(eval echo ${INPUT:-${ARENA_WS_DIR}})")

echo "installing ${ARENA_ROSNAV_REPO}:${ARENA_BRANCH} on ROS2 ${ARENA_ROS_VERSION} to ${ARENA_WS_DIR}"
sudo echo 'confirmed'
mkdir -p "$ARENA_WS_DIR"
cd "$ARENA_WS_DIR"

export INSTALLED=src/arena/arena-rosnav/.installed

# == remove ros problems ==
files=$((grep -l "/ros" /etc/apt/sources.list.d/* | grep -v "ros2") || echo '')

if [ -n "$files" ]; then
    echo "The following files can cause some problems to installer:"
    echo "$files"
    read -p "Do you want to delete these files? (Y/n) [Y]: " choice
    choice=${choice:-Y}

    if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
        sudo rm -f $files
        echo "Deleted $(echo $files)"
    fi
    unset choice
fi

# == python deps ==

# pyenv
if [ ! -d ~/.pyenv ]; then
  curl https://pyenv.run | bash
  echo 'export PYENV_ROOT="$HOME/.pyenv"'                                 >> ~/.bashrc
  echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"'  >> ~/.bashrc
  echo 'eval "$(pyenv init -)"'                                           >> ~/.bashrc
  source ~/.bashrc
fi

# Poetry
if ! which poetry ; then
  echo "Installing Poetry...:"
  curl -sSL https://install.python-poetry.org | python3 -
  if ! grep -q 'export PATH="$HOME/.local/bin"' ~/.bashrc; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
  fi
  $HOME/.local/bin/poetry config virtualenvs.in-project true
fi

# == compile ros ==


sudo add-apt-repository universe -y
sudo apt-get update || echo 0
sudo apt-get install -y curl

echo "Installing tzdata...:"
export DEBIAN_FRONTEND=noninteractive
sudo apt install -y tzdata libompl-dev
sudo dpkg-reconfigure --frontend noninteractive tzdata

# ROS
echo "Setting up ROS2 ${ARENA_ROS_VERSION}..."

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null


# for building python
echo "Installing Python deps..." 
sudo apt-get install -y build-essential python3-pip zlib1g-dev libffi-dev libssl-dev libbz2-dev libreadline-dev libsqlite3-dev liblzma-dev libncurses-dev tk-dev

if [ ! -d src/arena/arena-rosnav/tools ] ; then
  mkdir -p src/arena/arena-rosnav/tools
  pushd src/arena/arena-rosnav/tools
    curl "https://raw.githubusercontent.com/${ARENA_ROSNAV_REPO}/${ARENA_BRANCH}/tools/poetry_install" > poetry_install
    curl "https://raw.githubusercontent.com/${ARENA_ROSNAV_REPO}/${ARENA_BRANCH}/tools/colcon_build" > colcon_build
  popd
fi

if [ ! -d "${ARENA_WS_DIR}/src/arena/arena-rosnav/.venv" ] ; then
  #python env
  
  mkdir -p src/arena/arena-rosnav
  pushd src/arena/arena-rosnav
    curl "https://raw.githubusercontent.com/${ARENA_ROSNAV_REPO}/${ARENA_BRANCH}/pyproject.toml" > pyproject.toml
  popd
fi
. src/arena/arena-rosnav/tools/poetry_install

# vcstool fork (always reinstall)
if [ ! -d vcstool/.git ] ; then
  rm -f vcstool
  git clone https://github.com/voshch/vcstool.git vcstool
else
  pushd vcstool
    git pull
  popd
fi
python -m pip install -e vcstool

# Getting Packages
echo "Installing deps...:"
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    libasio-dev \
    libtinyxml2-dev \
    libcunit1-dev \
    ros-dev-tools \
    libpcl-dev

# Check if the default ROS sources.list file already exists
ros_sources_list="/etc/ros/rosdep/sources.list.d/20-default.list"
if [[ -f "$ros_sources_list" ]]; then
  echo "rosdep appears to be already initialized"
  echo "Default ROS sources.list file already exists:"
  echo "$ros_sources_list"
else
  sudo rosdep init
fi

rosdep update

if [ ! -d src/deps ] ; then
  #TODO resolve this through vcstool
  mkdir -p src/deps
  pushd src/deps
    git clone https://github.com/ros-perception/pcl_msgs.git -b ros2
    git clone https://github.com/rudislabs/actuator_msgs
    git clone https://github.com/swri-robotics/gps_umd
    git clone https://github.com/ros-perception/vision_msgs
    git clone https://github.com/ros-perception/vision_opencv.git -b "$ARENA_ROS_VERSION"
  popd
fi

if [ ! -f src/ros2/compiled ] ; then
  # install ros2

  mkdir -p src/ros2
  curl "https://raw.githubusercontent.com/ros2/ros2/${ARENA_ROS_VERSION}/ros2.repos" > ros2.repos
  vcs import src/ros2 < ros2.repos
  
  rosdep install \
    --from-paths src/ros2 \
    --ignore-src \
    --rosdistro "${ARENA_ROS_VERSION}" \
    -y \
    --skip-keys "fastcdr rti-connext-dds-6.0.1 urdfdom_headers" \
    || echo 'rosdep failed to install all dependencies'

  # fix rosidl error that was caused upstream https://github.com/ros2/rosidl/issues/822#issuecomment-2403368061
  pushd src/ros2/ros2/rosidl
    git cherry-pick 654d6f5658b59009147b9fad9b724919633f38fe || echo 'already cherry picked'
  popd

  . src/arena/arena-rosnav/tools/colcon_build --paths src/ros2/*
  touch src/ros2/compiled
fi

# == install arena on top of ros2 ==

if [ ! -f "$INSTALLED" ] ; then
  mv src/arena/arena-rosnav src/arena/.arena-rosnav

  echo "cloning Arena-Rosnav..."
  git clone --branch "${ARENA_BRANCH}" "https://github.com/${ARENA_ROSNAV_REPO}.git" src/arena/arena-rosnav

  mv -n src/arena/.arena-rosnav/* src/arena/arena-rosnav
  rm -rf src/arena/.arena-rosnav

  ln -fs src/arena/arena-rosnav/tools/poetry_install .
  ln -fs src/arena/arena-rosnav/tools/colcon_build .

  . poetry_install
fi

vcs import src < src/arena/arena-rosnav/arena.repos
rosdep install -y \
  --from-paths src/deps \
  --ignore-src \
  --rosdistro "$ARENA_ROS_VERSION" \
  || echo 'rosdep failed to install all dependencies'
touch "$INSTALLED"


#run installers
# sudo apt upgrade

compile(){
  rosdep install \
    --from-paths src \
    --ignore-src \
    --rosdistro ${ARENA_ROS_VERSION} \
    -y \
    --skip-keys "console_bridge fastcdr fastrtps libopensplice67 libopensplice69 rti-connext-dds-5.3.1 urdfdom_headers  DART libogre-next-2.3-dev transforms3d" \
    || echo 'rosdep failed to install all dependencies'
  cd "${ARENA_WS_DIR}"
  . colcon_build
}

compile

for installer in $(ls src/arena/arena-rosnav/installers | grep -E '^[0-9]+_.*.sh') ; do 

  name=$(echo $installer | cut -d '_' -f 2)

  if grep -q "$name" "$INSTALLED" ; then
    echo "$name already installed"
  else
    read -p "Do you want to install ${name}? [N] " choice
    choice="${choice:-N}"
    if [[ "$choice" =~ ^[Yy]$ ]]; then
        . "src/arena/arena-rosnav/installers/$installer"
        compile
        echo "$name" >> "$INSTALLED"
    else
        echo "Skipping ${name} installation."
    fi
    unset choice
  fi
done


# final pass
compile

echo 'installation finished'