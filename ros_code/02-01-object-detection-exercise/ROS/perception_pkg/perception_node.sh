#!/bin/bash
# Setup script for setting environment for team DiveDeep project

dir=/home/freicar/freicar_ws/src/ros_code/02-01-object-detection-exercise/ROS/perception_pkg/

source /opt/conda/etc/profile.d/conda.sh
conda activate freicar
pip install visdom
pip install webcolors

cd $dir
python perception_node.py -n $1 -w $2
