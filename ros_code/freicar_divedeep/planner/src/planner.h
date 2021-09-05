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

class planner{
public:
    planner(std::shared_ptr<ros::NodeHandle> n);
    std::shared_ptr<ros::NodeHandle> n_;

    void ExtControlCallback(const freicar_common::FreiCarControl::ConstPtr &ctrl_cmd);
    void CommandNewPlanner(bool goal_reach_flg, bool cmd_changed_flag);
    void GoalReachedStatus(const std_msgs::Bool reached);
    void GetParticles(const nav_msgs::Odometry msg);
    void GetParticles1(const geometry_msgs::PoseArray msg);

    std::vector<freicar::mapobjects::Point3D>OvertakePlan();

    ros::Subscriber sub;
    ros::Subscriber goal_reached,depth_info;
    ros::Subscriber freicar_commands;
    ros::Subscriber boundingbox_sub;
    ros::Subscriber request_overtake;

    ros::Publisher path_segment;
    ros::Publisher tf,overtake_plan;
    ros::Publisher stopline_status;
    ros::Publisher right_of_way_status;
    ros::Publisher Overtake_status;

    freicar::mapobjects::Point3D start_point ;
    std_msgs::Bool car_stop_status,standing_vehicle;
    std::string old_lane_uuid;

    float start_angle = 0.0;
    bool car_depth_ = false;
    bool Send_Overtake_plan = false;
    ros::Time time_when_last_stopped;
    ros::Time last_time_of_no_right_of_way;
    tf2_ros::Buffer tf_buffer_;

    std::string agent_name, map_name, map_path;
    float init_x, init_y, heading;
    bool use_yaml_spawn;

private:
    freicar::enums::PlannerCommand command = freicar::enums::PlannerCommand::STRAIGHT;
    std::string planner_cmd;
    bool goal_reached_flag;
    bool command_changed = false;
    std_msgs::Bool right_of_way;

    tf2_ros::TransformListener tf_obs_agent_listener;

    int findPathDescription(freicar::mapobjects::Lane::Connection description);

};


#endif //FREICAR_PLANNER_H