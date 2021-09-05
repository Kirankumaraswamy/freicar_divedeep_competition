#include "planner.h"

#include "ros/ros.h"
#include "std_msgs/String.h"
#include "geometry_msgs/PoseArray.h"
#include <freicar_map/planning/lane_star.h>
#include <freicar_map/planning/lane_follower.h>
#include <freicar_map/thrift_map_proxy.h>
#include "map_core/freicar_map.h"
#include "freicar_common/shared/planner_cmd.h"
#include <cstdio>
#include "nav_msgs/Path.h"
#include <visualization_msgs/MarkerArray.h>
#include "raiscar_msgs/ControllerPath.h"

planner::planner(std::shared_ptr<ros::NodeHandle> n): n_(n), tf_obs_agent_listener(tf_buffer_)
{
    n_->param<std::string>("agent_name", agent_name, "freicar_1");
    n_->param<bool>("use_yaml_spawn", use_yaml_spawn, true);
    //n_->param<float>("init_x", init_x, 1.0);
    //n_->param<float>("init_y", init_y, 0);
    n_->param<float>("heading", heading, 0);
    n_->param<std::string>("map_name", map_name, "freicar_1.aismap");
    //n_->param<std::string>("map_path", map_path, "map_path");
    std::cout <<agent_name <<" " <<use_yaml_spawn <<" " <<init_x <<" " <<init_y <<" " <<heading <<" "<<map_name <<std::endl;


    sub = n_->subscribe(agent_name+"/best_particle", 10, &planner::GetParticles1, this);
    //sub = n_->subscribe(agent_name+"/odometry", 10, &planner::GetParticles, this);
    goal_reached = n_->subscribe(agent_name+"/goal_reached", 10, &planner::GoalReachedStatus, this);
    //request_overtake = n_->subscribe("freicar_1/request_overtake",1, &plan_publisher::RequestOvertakeStatus,this);
    //depth_info = n_->subscribe("/car_ahead", 1, &plan_publisher::DepthInfoStatus, this);

    freicar_commands = n_->subscribe("/freicar_commands",5 , &planner::ExtControlCallback, this);
    //boundingbox_sub = n_->subscribe("/bbsarray", 1, &plan_publisher::BoundingBoxCallback, this);
    path_segment = n_->advertise<raiscar_msgs::ControllerPath>(agent_name+"/path_segment", 10);
    //path_segment = n_->advertise<raiscar_msgs::ControllerPath>("/freicar_1/path_segment", 10);
    //stopline_status = n_->advertise<std_msgs::Bool>("stopline_status", 1);
    //right_of_way_status = n_->advertise<std_msgs::Bool>("right_of_way", 1);
    //Overtake_status = n_->advertise<std_msgs::Bool>("Standing_Vehicle", 1);

    //tf = n_->advertise<visualization_msgs::MarkerArray>("planner_debug", 10, true);
    //right_of_way.data = true;
    //overtake_plan = n_->advertise<visualization_msgs::MarkerArray>("overtake_planner", 10, true);

}

void planner::GoalReachedStatus(const std_msgs::Bool reached) {
    goal_reached_flag = reached.data;
    if(!command_changed){
        CommandNewPlanner(goal_reached_flag, command_changed);
    }
    goal_reached_flag = false; //reset the flag
}

void planner::ExtControlCallback(const freicar_common::FreiCarControl::ConstPtr &ctrl_cmd)
{
    if(ctrl_cmd->name == agent_name){
        goal_reached_flag = false;
        planner_cmd = ctrl_cmd->command;
        if(planner_cmd == "right")
        {
            command = freicar::enums::PlannerCommand::RIGHT;
            command_changed = true;
        }
        else if(planner_cmd == "left")
        {
            command = freicar::enums::PlannerCommand::LEFT;
            command_changed = true;
        }
        else if(planner_cmd == "right")
        {
            command = freicar::enums::PlannerCommand::RIGHT;
            command_changed = true;
        }
        else if(planner_cmd == "straight")
        {
            command = freicar::enums::PlannerCommand::STRAIGHT;
            command_changed = true;
        }

        if(!goal_reached_flag) {
            CommandNewPlanner(goal_reached_flag, command_changed);
        }
    }
    else{
        command_changed = false; //command changes flag reset
    }

}

//Listens to higher level commands and plans accordingly
void planner::CommandNewPlanner(bool goal_reach_flg, bool cmd_changed_flag) {
    if(goal_reach_flg || cmd_changed_flag){
        geometry_msgs::PoseStamped pose_msg;
        raiscar_msgs::ControllerPath rais_control_msg;
        ROS_INFO("PATH Published .....");
        auto lane_plan = freicar::planning::lane_follower::GetPlan(start_point, command, 8, 40);

        // To publish on RVIZ
        //PublishPlan(lane_plan, 0.0, 1.0, 0.0, 300, "plan_1", tf);

        // To publish on ROS
        rais_control_msg.path_segment.header.stamp = ros::Time::now();
        rais_control_msg.path_segment.header.frame_id = "map";
        rais_control_msg.path_segment.header.seq = 0;

        // To publish on ROS
        for(size_t i = 0; i < lane_plan.steps.size(); ++i) {
            pose_msg.pose.position.x = lane_plan.steps[i].position.x();
            pose_msg.pose.position.y = lane_plan.steps[i].position.y();
            pose_msg.pose.position.z = lane_plan.steps[i].position.z();
            /* Orientation used as a proxy for sending path description */
            //pose_msg.pose.orientation.w = findPathDescription(lane_plan.steps[i].path_description);
            rais_control_msg.path_segment.poses.push_back(pose_msg);
        }
        path_segment.publish(rais_control_msg);

    }
}

int planner::findPathDescription(freicar::mapobjects::Lane::Connection description) {
    /*
     * 0 = JUNCTION_STRAIGHT: The next lane in a junction that goes straight
     * 1 = JUNCTION_LEFT: The next lane in a junction that turns left
     * 2 = JUNCTION_RIGHT: The next lane in a junction that turns right
     * 3 = STRAIGHT: The next lane that's not part of a junction
     * 4 = OPPOSITE: The opposite lane
     * 5 = ADJACENT_LEFT: The adjacent lane to the left
     * 6 = ADJACENT_RIGHT: The adjacent lane to the left
     * 7 = BACK: The previous lane
     * */
    int converted;
    switch(description){
        case freicar::mapobjects::Lane::JUNCTION_STRAIGHT:
            converted = 0;
            break;
        case freicar::mapobjects::Lane::JUNCTION_LEFT:
            converted = 1;
            break;
        case freicar::mapobjects::Lane::JUNCTION_RIGHT:
            converted = 2;
            break;
        case freicar::mapobjects::Lane::STRAIGHT:
            converted = 3;
            break;
        case freicar::mapobjects::Lane::OPPOSITE:
            converted = 4;
            break;
        case freicar::mapobjects::Lane::ADJACENT_LEFT:
            converted = 5;
            break;
        case freicar::mapobjects::Lane::ADJACENT_RIGHT:
            converted = 6;
            break;
        case freicar::mapobjects::Lane::BACK:
            converted = 7;
            break;
        default:
            converted = 3;
    }
    return converted;
}

// Call back function to get the current location of the car.
void planner::GetParticles(const nav_msgs::Odometry msg){
    // Fetching the current location of the car which is plan's starting point
    freicar::mapobjects::Point3D current_point(msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z);
    start_point = current_point;
    std::cout << "Odometry: " <<  msg.pose.pose.position.x <<", "<< msg.pose.pose.position.y << std::endl;
    /*double x_ang = msg.pose.pose.position.x;
    double y_ang = msg.pose.pose.position.y;
    double z_ang = msg.pose.pose.position.z;
    double w_ang = msg.pose.pose.orientation.w;
    start_angle = atan2(2.0f * (w_ang * z_ang + x_ang * y_ang), 1.0f - 2.0f * (y_ang * y_ang + z_ang * z_ang));

    // To Fetch the current lane.
    const freicar::mapobjects::Lane *current_lane;
    auto &map = freicar::map::Map::GetInstance();
    auto p_closest = map.FindClosestLanePoints(current_point.x(), current_point.y(), current_point.z(), 1)[0].first;
    auto current_lane_uuid = p_closest.GetLaneUuid();
    current_lane = map.FindLaneByUuid(current_lane_uuid);

    // To Fetch the sign boards connected to the current lane.
    freicar::mapobjects::Point3D stoplinepos;
    float distance;
    // If stop sign detected.
    if(current_lane->HasRoadSign("Stop")) {
        const freicar::mapobjects::Stopline *stopline = current_lane->GetStopLine();
        stoplinepos = stopline->GetPosition();
        distance = stoplinepos.ComputeDistance(current_point);
        if(distance < 0.7){
            car_stop_status.data = true;
        }
    }
        // No stop sign detected.
    else {
        car_stop_status.data = false;
    }
    ros::Time now = ros::Time::now();
    // If car needed to be stopped and for a different lane than before.
    if(car_stop_status.data == true) {
        if (current_lane_uuid != old_lane_uuid) {
            old_lane_uuid = current_lane_uuid;
            stopline_status.publish(car_stop_status);
            time_when_last_stopped = ros::Time::now();
            car_stop_status.data = false;
        }
        else if((current_lane_uuid == old_lane_uuid) && ((now-time_when_last_stopped).toSec() > 10)){
            old_lane_uuid = current_lane_uuid;
            stopline_status.publish(car_stop_status);
            time_when_last_stopped = ros::Time::now();
            car_stop_status.data = false;
        }
    }*/
}

// Call back function to get the current location of the car.
void planner::GetParticles1(const geometry_msgs::PoseArray msg){
    // Fetching the current location of the car which is plan's starting point
    freicar::mapobjects::Point3D current_point(msg.poses.data()->position.x, msg.poses.data()->position.y, msg.poses.data()->position.z);
    start_point = current_point;
    std::cout << "Localization: " <<  msg.poses.data()->position.x <<", "<< msg.poses.data()->position.y << std::endl;
}

int main(int argc, char **argv)
{

    ros::init(argc, argv, "planner");

    freicar::map::ThriftMapProxy map_proxy("127.0.0.1", 9091, 9090);
    std::string map_path;

    if (!ros::param::get("/map_path", map_path)) {
        //map_path = "/home/freicar/freicar_ws/src/freicar_base/freicar_map/maps/freicar_1.aismap";
        ROS_ERROR("could not find parameter: map_path! map initialization failed");
        return 0;
    }

// if the map can't be loaded
    if (!map_proxy.LoadMapFromFile(map_path)) {
        ROS_INFO("could not find thriftmap file: %s, starting map server...", map_path.c_str());
        map_proxy.StartMapServer();
        // stalling main thread until map is received
        while (freicar::map::Map::GetInstance().status() == freicar::map::MapStatus::UNINITIALIZED) {
//            ROS_INFO("waiting for map...", );
            ros::Duration(1.0).sleep();
        }
        ROS_INFO("map received!");
        // Thrift creats a corrupt file on failed reads, removing it
        remove(map_path.c_str());
        map_proxy.WriteMapToFile(map_path);
        ROS_INFO("saved new map");
    }
    freicar::map::Map::GetInstance().PostProcess(0.22);

    std::shared_ptr<ros::NodeHandle> n_ = std::make_shared<ros::NodeHandle>();
    planner plan(n_);


    ros::spin();

    return 0;
}