# Instructions to run ROS node

Take a clone of the 02-02-semantic-segmentation-exercise repository from the link 'https://github.com/madhu-basavanna/Freicar_DiveDeep/edit/master/02-02-semantic-segmentation-exercise'

Run catkin build from the terminal at file location ~/freicar_ws you will find the build of 'ros_pkg' package under ROS folder

Run 'roslaunch freicar_launch local_comp_launch.launch' from the terminal to launch the simulator

Run 'roslaunch freicar_agent sim_agent.launch name:=freicar_1 tf_name:=freicar_1 spawn/x:=0 spawn/y:=0 spawn/z:=0 spawn/heading:=20 use_yaml_spawn:=true sync_topic:=!' to spawn the car

Run 'rosrun freicar_executables freicar_carla_agent_node 3'

Change the directory in the terminal to ~/02-02-semantic-segmentation-exercise/ROS/ros_pkg

Activate freicar python environment

Run 'python lane_detector.py --load_model recent_model_19.pth.tar' from the terminal

The 'lane_detector' node is subscribed to '/freicar_1/sim/camera/rgb/front/image', which receives the input image from camera and passes to the model to perform the inference.

The Semantic Segmentation and Birds Eye View of Lane Regression images are published from node 'lane_detector' to topics 'semantic_segmentation' and 'lane_detection' respectively.

Subscribe to topic '/lreg' in rviz to visualise the MarkerArray
