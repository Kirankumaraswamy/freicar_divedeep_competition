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


    //sub = n_->subscribe(agent_name+"/best_particle", 10, &planner::InitializeBestParticle, this);
    //sub = n_->subscribe(agent_name+"/odometry", 10, &planner::InitializeBestParticle1, this);
    sub = n_->subscribe("car_localization", 10, &planner::InitializeBestParticle2, this);
    goal_reached_a = n_->subscribe(agent_name+"/goal_reached", 10, &planner::GoalReachedStatusReceived, this);

    freicar_commands = n_->subscribe("/freicar_commands",10 , &planner::ExecuteCommand, this);
    //boundingbox_sub = n_->subscribe("/bbsarray", 1, &plan_publisher::BoundingBoxCallback, this);
    path_segment = n_->advertise<raiscar_msgs::ControllerPath>(agent_name+"/path_segment", 10);
    std::cout << "Initialize ....." << std::endl;

    tf = n_->advertise<visualization_msgs::MarkerArray>("planner_debug", 10, true);
    //right_of_way.data = true;
    //overtake_plan = n_->advertise<visualization_msgs::MarkerArray>("overtake_planner", 10, true);

}

void PublishPlan (freicar::planning::Plan& plan, double r, double g, double b, int id, const std::string& name, ros::Publisher& pub) {
    visualization_msgs::MarkerArray list;
    visualization_msgs::Marker *step_number = new visualization_msgs::Marker[plan.size()];
    int num_count = 0;
    visualization_msgs::Marker plan_points;
    plan_points.id = id;
    plan_points.ns = name;
    plan_points.header.stamp = ros::Time();
    plan_points.header.frame_id = "map";
    plan_points.action = visualization_msgs::Marker::ADD;
    plan_points.type = visualization_msgs::Marker::POINTS;
    plan_points.scale.x = 0.03;
    plan_points.scale.y = 0.03;
    plan_points.pose.orientation = geometry_msgs::Quaternion();
    plan_points.color.b = b;
    plan_points.color.a = 0.7;
    plan_points.color.g = g;
    plan_points.color.r = r;
    geometry_msgs::Point p;
    for (size_t i = 0; i < plan.size(); ++i) {
        step_number[i].id = ++num_count + id;
        step_number[i].pose.position.x = p.x = plan[i].position.x();
        step_number[i].pose.position.y = p.y = plan[i].position.y();
        p.z = plan[i].position.z();
        step_number[i].pose.position.z = plan[i].position.z() + 0.1;
        step_number[i].pose.orientation = geometry_msgs::Quaternion();
        step_number[i].ns = name + "_nums";
        step_number[i].header.stamp = ros::Time();
        step_number[i].header.frame_id = "map";
        step_number[i].action = visualization_msgs::Marker::ADD;
        step_number[i].type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        step_number[i].text = std::to_string(i);
        step_number[i].scale.z = 0.055;
        step_number[i].color = plan_points.color;
        list.markers.emplace_back(step_number[i]);
        plan_points.points.emplace_back(p);
    }
    list.markers.emplace_back(plan_points);
    pub.publish(list);
    delete[] step_number;
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
    else{
        command_changed = false; //command changes flag reset
    }

}

void planner::PublishNewPlan() {
    geometry_msgs::PoseStamped pose_msg;
    raiscar_msgs::ControllerPath rais_control_msg;

    auto plan = freicar::planning::lane_follower::GetPlan(current_position, command, 8, 80);

    PublishPlan(plan, 0.0, 1.0, 0.0, 300, "plan_1", tf);

    rais_control_msg.path_segment.header.stamp = ros::Time::now();
    rais_control_msg.path_segment.header.frame_id = "map";
    rais_control_msg.path_segment.header.seq = 0;

    for(size_t i = 0; i < plan.steps.size(); ++i) {
        pose_msg.pose.position.x = plan.steps[i].position.x();
        pose_msg.pose.position.y = plan.steps[i].position.y();
        pose_msg.pose.position.z = plan.steps[i].position.z();

        //pose_msg.pose.orientation.w = plan.steps[i].path_description;
        pose_msg.pose.orientation.w = findPathDescription(plan.steps[i].path_description);
        std::cout << i << " " << plan.steps[i].position.x()<< " "<< plan.steps[i].position.y()<< " " << plan.steps[i].path_description << " " << pose_msg.pose.orientation.w<<std::endl;
        rais_control_msg.path_segment.poses.push_back(pose_msg);
    }
    path_segment.publish(rais_control_msg);
    std::cout << "Path Published ....." << command <<std::endl;
    ROS_INFO("PATH published...................");
    //reset the command to straight
    command = freicar::enums::PlannerCommand::STRAIGHT;
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
