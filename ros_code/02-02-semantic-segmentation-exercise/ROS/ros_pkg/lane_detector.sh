#!/bin/bash

dir=/home/freicar/freicar_ws/src/ros_code/02-02-semantic-segmentation-exercise/ROS/ros_pkg

source /opt/conda/etc/profile.d/conda.sh
conda activate freicar

cd $dir
python lane_detector.py -n $1 -w $2
