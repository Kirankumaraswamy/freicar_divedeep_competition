
# Exercise 1, Control, Pure Pursuit Controller

## How to run the code:
## To launch the simulator
Run `roslaunch freicar_launch local_comp_launch.launch`.

## To spawn the CAR
Run `roslaunch freicar_agent sim_agent.launch name:=freicar_1 tf_name:=freicar_1 spawn/x:=0 spawn/y:=0 spawn/z:=0 spawn/heading:=20 use_yaml_spawn:=true sync_topic:=!`.

## How to start the node
Run `roslaunch freicar_control start_controller.launch`.

## How to send a path
Run `rosrun freicar_control pub_path.py`. This node will send a predefined path to the controller and additionally publishes it as PoseArray message that can be visualized using Rviz.

## Plot the graph
Run `plot.py` python file.
