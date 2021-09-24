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
#include "std_msgs/Float32MultiArray.h"
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
#include "std_msgs/Float32.h"
#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>

class planner{
public:
    planner(std::shared_ptr<ros::NodeHandle> n);
    std::shared_ptr<ros::NodeHandle> n_;

    void ExecuteCommand(const freicar_common::FreiCarControl::ConstPtr &ctrl_cmd);
    void PublishNewPlan();
    void PublishOvertakePlan();
    void GoalReachedStatusReceived(const std_msgs::Bool reached);
    void InitializeBestParticle1(const nav_msgs::Odometry msg);
    void InitializeBestParticle2(const freicar_common::FreiCarAgentLocalization msg);
    void InitializeBestParticle(const geometry_msgs::PoseArray msg);
    void DepthCallback(const sensor_msgs::ImageConstPtr& msg);
    void BoundingBoxRecieved(const std_msgs::Float32MultiArray msg);
    float check_collision(Eigen::Vector3f obj_point, float px, float py);
    void SetOvertakeStatus(const std_msgs::Bool status);

    std::vector<freicar::mapobjects::Point3D>OvertakePlan();
    ros::Subscriber freicar_commands;

    freicar::mapobjects::Point3D current_position ;
    float current_angle;

    ros::Subscriber sub;
    ros::Subscriber goal_reached_a,depth_info, rgb, depth;
    ros::Subscriber external_control_sub;
    ros::Subscriber boundingbox_sub;
    ros::Subscriber overtake_pub;


    ros::Publisher path_segment;
    ros::Publisher otpath_segment;
    ros::Publisher broadcast,overtake_plan;
    ros::Publisher stopline_status;
    ros::Publisher right_of_way_status;
    ros::Publisher obstacle_status;
    ros::Publisher stop_status;
    ros::Publisher pub_overtake_permission;
    std_msgs::Float32 obstacleDistance;

    ros::Publisher marker_pub;

    float obstacle_radius;

    ros::Time last_stop_publish_time;
    std_msgs::Float32 stopDistance;
    std_msgs::Float32MultiArray current_boundig_bxs;
    cv::Mat depth_image;
    bool overtake_status;

    std::string agent_name, map_name, map_path;
    float init_x, init_y, heading;
    bool use_yaml_spawn;
    tf2_ros::Buffer tf_buffer_;
    std_msgs::Float32MultiArray bbs;

    std::string closest_lane_uuid;

    std::vector<Eigen::Vector3f> published_path;

    freicar::enums::PlannerCommand command = freicar::enums::PlannerCommand::STRAIGHT;
    bool goal_reached, publish_empty_plan, activate_overtake_plan;
    bool command_changed = false;
    tf2_ros::TransformListener tf_obs_agent_listener;
    int getDirectionValues(freicar::mapobjects::Lane::Connection value);
};


#endif //FREICAR_PLANNER_H
