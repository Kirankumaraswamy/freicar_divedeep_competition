# General note: Make sure you do not use hardcoded paths in your code ever!
# Instead you should always make paths relative to some ros-package. 
# Refer to https://wiki.ros.org/Packages#Client_Library_Support to see how to get paths of packages.

### DO NOT CHANGE THIS PART, ONLY AFTER ############################################################################
car_name=$1
use_yaml_spawn=$2
x_spawn=$3
y_spawn=$4
yaw_spawn=$5

# collect additional commands
add_commands=
counter=0
for i in "$@"
do
  counter=$((counter+1))
  if [[ $counter -gt 5 ]]
  then
    add_commands="$add_commands $i"
  fi
done

echo "starting car: $car_name"

source /opt/ros/noetic/setup.bash
source /home/freicar/freicar_ws/devel/setup.bash
####################################################################################################################

# Your launch file goes here! Note: You can only start one launch file in this script. However you can include additional launch files in this launch file.
# Make sure you consider the car_name and all spawn variables
roslaunch /home/freicar/freicar_ws/src/ros_code/freicar.launch agent_name:=$car_name use_yaml_spawn:=$use_yaml_spawn init_x:=$x_spawn init_y:=$y_spawn heading:=$yaw_spawn ${add_commands}
#roslaunch freicar_agent sim_agent.launch name:=$car_name tf_name:=$car_name spawn/x:=$x_spawn spawn/y:=$y_spawn spawn/z:=0 spawn/heading:=$yaw_spawn use_yaml_spawn:=$use_yaml_spawn sync_topic:=! ${add_commands# }

