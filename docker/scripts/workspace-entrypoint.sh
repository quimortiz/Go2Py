#!/bin/bash
#
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Build ROS dependency
echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
source /opt/ros/${ROS_DISTRO}/setup.bash

sudo apt-get update
rosdep update

# Restart udev daemon
sudo service udev restart
export CMAKE_PREFIX_PATH=/workspace/unitree_ros2/cyclonedds_ws/install/cyclonedds/
python3 -m pip install --upgrade pip
cd /workspaces/Go2Py/ && python3 -m pip install -e .
source /home/realsenes-ws/install/setup.bash
$@
