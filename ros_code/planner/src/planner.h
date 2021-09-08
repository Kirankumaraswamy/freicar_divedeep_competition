#ifndef FREICAR_PLANNER_H
#define FREICAR_PLANNER_H

#include <string>
#include <cmath>
#include <algorithm>
#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/transform_datatypes.h>
#include <tf2/transform_storage.h>
#include <tf2/buffer_core.h>
#include <tf2/convert.h>
#include <tf2/utils.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include "raiscar_msgs/ControlCommand.h"
#include "std_msgs/Bool.h"
#include "raiscar_msgs/ControllerPath.h"
#include <freicar_map/thrift_map_proxy.h>

#include "freicar_common/FreiCarControl.h"


#include "ros/ros.h"
#include <geometry_msgs/PoseArray.h>
#include "std_msgs/String.h"
#include <freicar_map/planning/lane_star.h>
#include <freicar_map/planning/lane_follower.h>
#include <freicar_map/thrift_map_proxy.h>
#include "map_core/freicar_map.h"
#include "freicar_common/shared/planner_cmd.h"
#include <cstdio>
#include <visualization_msgs/MarkerArray.h>
#include "nav_msgs/Path.h"
#include "std_msgs/Bool.h"
#include <tf2_ros/transform_listener.h>
#include <tf2/transform_datatypes.h>
#include <tf2/utils.h>
#include <tf2/convert.h>

#include "freicar_common/FreiCarControl.h"
#include <cstdio>
#include "nav_msgs/Path.h"
#include <visualization_msgs/MarkerArray.h>
#include "raiscar_msgs/ControllerPath.h"
#include "tf2/transform_datatypes.h"
#include "tf2/transform_storage.h"
#include <tf2/convert.h>
#include <tf2/utils.h>
#include "Eigen/Dense"
#include "freicar_common/FreiCarAgentLocalization.h"

class planner{
public:
    planner(std::shared_ptr<ros::NodeHandle> n);
    std::shared_ptr<ros::NodeHandle> n_;

    void ExecuteCommand(const freicar_common::FreiCarControl::ConstPtr &ctrl_cmd);
    void PublishNewPlan();
    void GoalReachedStatusReceived(const std_msgs::Bool reached);
    void InitializeBestParticle1(const nav_msgs::Odometry msg);
    void InitializeBestParticle2(const freicar_common::FreiCarAgentLocalization msg);
    void InitializeBestParticle(const geometry_msgs::PoseArray msg);

    std::vector<freicar::mapobjects::Point3D>OvertakePlan();
    ros::Subscriber freicar_commands;

    freicar::mapobjects::Point3D current_position ;

    ros::Subscriber sub;
    ros::Subscriber goal_reached_a,depth_info;
    ros::Subscriber external_control_sub;
    ros::Subscriber boundingbox_sub;
    ros::Subscriber request_overtake;


    ros::Publisher path_segment;
    ros::Publisher tf,overtake_plan;
    ros::Publisher stopline_status;
    ros::Publisher right_of_way_status;
    ros::Publisher Overtake_status;

    std::string agent_name, map_name, map_path;
    float init_x, init_y, heading;
    bool use_yaml_spawn;
    tf2_ros::Buffer tf_buffer_;

    freicar::enums::PlannerCommand command = freicar::enums::PlannerCommand::STRAIGHT;
    bool goal_reached;
    bool command_changed = false;
    tf2_ros::TransformListener tf_obs_agent_listener;
    int getDirectionValues(freicar::mapobjects::Lane::Connection value);
};


#endif //FREICAR_PLANNER_H
