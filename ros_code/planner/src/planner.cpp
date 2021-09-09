#include "planner.h"

#include "ros/ros.h"
#include "std_msgs/String.h"
#include "geometry_msgs/PoseArray.h"
#include <freicar_map/planning/lane_follower.h>
#include <freicar_map/thrift_map_proxy.h>
#include "map_core/freicar_map.h"
#include "freicar_common/shared/planner_cmd.h"
#include <cstdio>
#include "nav_msgs/Path.h"
#include "raiscar_msgs/ControllerPath.h"
#include "freicar_common/FreiCarAgentLocalization.h"

planner::planner(std::shared_ptr<ros::NodeHandle> n): n_(n), tf_obs_agent_listener(tf_buffer_)
{
    n_->param<std::string>("agent_name", agent_name, "freicar_1");
    n_->param<std::string>("map_name", map_name, "freicar_1.aismap");


    sub = n_->subscribe(agent_name+"/best_particle", 10, &planner::InitializeBestParticle, this);
    //sub = n_->subscribe(agent_name+"/odometry", 10, &planner::InitializeBestParticle1, this);
    //sub = n_->subscribe("car_localization", 10, &planner::InitializeBestParticle2, this);
    goal_reached_a = n_->subscribe(agent_name+"/goal_reached", 10, &planner::GoalReachedStatusReceived, this);

    freicar_commands = n_->subscribe("/freicar_commands",10 , &planner::ExecuteCommand, this);
    path_segment = n_->advertise<raiscar_msgs::ControllerPath>(agent_name+"/path_segment", 10);
    std::cout << "Initialize ....." << std::endl;

}

void planner::GoalReachedStatusReceived(const std_msgs::Bool goal_message) {
    goal_reached = goal_message.data;
    command_changed = false;
    if(goal_reached == true){
        PublishNewPlan();
    }
    goal_reached = false;

    std::cout << "Goal Reached ....." << std::endl;
}

void planner::ExecuteCommand(const freicar_common::FreiCarControl::ConstPtr &control_command)
{
    std::cout << "Command Reached ....." << std::endl;
    if(control_command->name == agent_name){
        command_changed = true;
        goal_reached = false;
        std::string cmd = control_command->command;

        if(cmd == "start")        {
            command = freicar::enums::PlannerCommand::STRAIGHT;
        }
        else if(cmd == "left")        {
            command = freicar::enums::PlannerCommand::LEFT;
        }
        else if(cmd == "right")        {
            command = freicar::enums::PlannerCommand::RIGHT;
        }
        else if(cmd == "straight")
        {
            command = freicar::enums::PlannerCommand::STRAIGHT;
        }
        PublishNewPlan();
    }
    else {
        command_changed = false;
    }

}

void planner::PublishNewPlan() {
    geometry_msgs::PoseStamped pose_msg;
    raiscar_msgs::ControllerPath rais_control_msg;

    auto plan = freicar::planning::lane_follower::GetPlan(current_position, command, 8, 80);
    rais_control_msg.path_segment.header.stamp = ros::Time::now();
    rais_control_msg.path_segment.header.frame_id = "map";
    rais_control_msg.path_segment.header.seq = 0;

    for(size_t i = 0; i < plan.steps.size(); ++i) {
        pose_msg.pose.position.x = plan.steps[i].position.x();
        pose_msg.pose.position.y = plan.steps[i].position.y();
        pose_msg.pose.position.z = plan.steps[i].position.z();

        //pose_msg.pose.orientation.w = plan.steps[i].path_description;
        pose_msg.pose.orientation.w = getDirectionValues(plan.steps[i].path_description);
        std::cout << i << " " << plan.steps[i].position.x()<< " "<< plan.steps[i].position.y()<< " " << plan.steps[i].path_description << " " << pose_msg.pose.orientation.w<<std::endl;
        rais_control_msg.path_segment.poses.push_back(pose_msg);
    }
    path_segment.publish(rais_control_msg);
    std::cout << "Path Published ....." << command <<std::endl;
    ROS_INFO("PATH published...................");
    //reset the command to straight
    command = freicar::enums::PlannerCommand::STRAIGHT;
}



int planner::getDirectionValues(freicar::mapobjects::Lane::Connection key) {
    int value;
    switch(key){
        case freicar::mapobjects::Lane::JUNCTION_STRAIGHT:
            value = 0;
            break;
        case freicar::mapobjects::Lane::JUNCTION_LEFT:
            value = 1;
            break;
        case freicar::mapobjects::Lane::JUNCTION_RIGHT:
            value = 2;
            break;
        case freicar::mapobjects::Lane::STRAIGHT:
            value = 3;
            break;
        case freicar::mapobjects::Lane::OPPOSITE:
            value = 4;
            break;
        case freicar::mapobjects::Lane::ADJACENT_LEFT:
            value = 5;
            break;
        case freicar::mapobjects::Lane::ADJACENT_RIGHT:
            value = 6;
            break;
        case freicar::mapobjects::Lane::BACK:
            value = 7;
            break;
        default:
            value = 3;
    }
    return value;
}


void planner::InitializeBestParticle1(const nav_msgs::Odometry msg){
    freicar::mapobjects::Point3D current_point(msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z);
    current_position = current_point;
    std::cout << "Odometry: " <<  msg.pose.pose.position.x <<", "<< msg.pose.pose.position.y << std::endl;
}

void planner::InitializeBestParticle2(const freicar_common::FreiCarAgentLocalization msg){
    freicar::mapobjects::Point3D current_point(msg.current_pose.transform.translation.x, msg.current_pose.transform.translation.y, msg.current_pose.transform.translation.z);
    current_position = current_point;
    std::cout << "Car Local: " <<  msg.current_pose.transform.translation.x <<", "<< msg.current_pose.transform.translation.y << std::endl;
}

void planner::InitializeBestParticle(const geometry_msgs::PoseArray msg){
    freicar::mapobjects::Point3D current_point(msg.poses.data()->position.x, msg.poses.data()->position.y, msg.poses.data()->position.z);
    current_position = current_point;
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
