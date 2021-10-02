#include "planner.h"
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Float32.h"
#include "geometry_msgs/PoseArray.h"
#include <freicar_map/planning/lane_follower.h>
#include <freicar_map/thrift_map_proxy.h>
#include "map_core/freicar_map.h"
#include "freicar_common/shared/planner_cmd.h"
#include <cstdio>
#include "nav_msgs/Path.h"
#include "raiscar_msgs/ControllerPath.h"
#include "freicar_common/FreiCarAgentLocalization.h"
#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"

planner::planner(std::shared_ptr<ros::NodeHandle> n): n_(n), tf_obs_agent_listener(tf_buffer_)
{
    n_->param<std::string>("agent_name", agent_name, "greatteam");
    n_->param<std::string>("map_name", map_name, "freicar_1.aismap");
    n_->param<float>("obstacle_radius", obstacle_radius, 0.4);

    sub = n_->subscribe(agent_name+"/best_particle", 1, &planner::InitializeBestParticle, this);
    //sub = n_->subscribe(agent_name+"/odometry", 10, &planner::InitializeBestParticle1, this);
    //sub = n_->subscribe("car_localization", 1, &planner::InitializeBestParticle2, this);
    goal_reached_a = n_->subscribe(agent_name+"/goal_reached", 1, &planner::GoalReachedStatusReceived, this);
    boundingbox_sub = n_->subscribe(agent_name+"/predicted_bbs", 1, &planner::BoundingBoxRecieved, this);
    request_overtake = n_->subscribe(agent_name+"/overtake_permission", 1, &planner::SetOvertakeStatus, this);

    freicar_commands = n_->subscribe("/freicar_commands",1 , &planner::ExecuteCommand, this);
    path_segment = n_->advertise<raiscar_msgs::ControllerPath>(agent_name+"/path_segment", 10);
    stop_status = n_->advertise<std_msgs::Float32>(agent_name+"/stop_signal", 1);
    obstacle_status = n_->advertise<std_msgs::Float32>(agent_name+"/obstacle_signal", 1);
    std::cout << "Initialize ....." << std::endl;

    broadcast = n_->advertise<visualization_msgs::MarkerArray>("planner", 10, true);
    marker_pub = n_->advertise<visualization_msgs::Marker>("object_detection", 1);

    last_stop_publish_time = ros::Time::now();
    send_obstacle = true;

}

void Overtake_PlanPublish (std::vector<Eigen::Vector3f> plan3dps, double r, double g, double b, int id, const std::string& name, ros::Publisher& pub) {
    std::cout << "******************Welcome to Overtake PlanPublish************************" << std::endl;
    visualization_msgs::MarkerArray list;
    visualization_msgs::Marker *step_number = new visualization_msgs::Marker[plan3dps.size()];
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
    for (size_t i = 0; i < plan3dps.size(); ++i) {
        step_number[i].id = ++num_count + id;
        step_number[i].pose.position.x = p.x = plan3dps[i].x();
        step_number[i].pose.position.y = p.y = plan3dps[i].y();
        p.z = plan3dps[i].z();
        step_number[i].pose.position.z = plan3dps[i].z() + 0.1;
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

void planner::SetOvertakeStatus(const std_msgs::Bool status) {
    overtake_status = status.data;
    if(overtake_status){
            PublishOvertakePlan();
    }
    std::cout << "Overtake Status" << overtake_status << std::endl;

    // overtake_status = false;
    // std::cout << "Overtake Status" << overtake_status << std::endl;
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

void planner::PublishOvertakePlan(){
    std::cout << "***************Overtake*******************" << obstacleDistance.data  << std::endl;
    geometry_msgs::PoseStamped pose_msg;
    raiscar_msgs::ControllerPath rais_control_msg;
    OvertakePlan();

}


void planner::OvertakePlan(){

    geometry_msgs::PoseStamped pose_msg;
    raiscar_msgs::ControllerPath rais_control_msg;
    //    const freicar::mapobjects::Lane* current_lane;
    std::cout << "Plan Overtake...................... " << obstacleDistance.data << std::endl;
    auto& getInstance = freicar::map::Map::GetInstance();
    const freicar::mapobjects::Lane* opposite_lane;

    std::vector<std::pair<freicar::mapobjects::LanePoint3D, float>> current_lane_points =
            getInstance.FindClosestLanePointsWithHeading(current_position.x(), current_position.y(), current_position.z(), 50, current_angle);

    auto current_lane = getInstance.FindLaneByUuid(current_lane_points[0].first.GetLaneUuid());
    auto* lanes = getInstance.FindLaneGroupByUuid(current_lane->GetParentUuid());

    std::vector<Eigen::Vector3f> published_path_temp;
    //deep copy published path
    for(size_t i = 0; i < published_path.size(); ++i) {
        published_path_temp.push_back(Eigen::Vector3f(published_path.at(i).x(), published_path.at(i).y(), published_path.at(i).z()));
    }

    //Check for junction
    if(!current_lane->IsJunctionLane()) {
        const freicar::mapobjects::Lane* opposite_lane;
        //For right lane
        if(current_lane->GetLaneDirection() == 1) {
            for (auto& left_uuid : lanes->GetLeftLanes()) {
                opposite_lane = getInstance.FindLaneByUuid(left_uuid);
            }
        }else if (current_lane->GetLaneDirection() == 2) {
            //for left lane
            for (auto &right_uuid : lanes->GetRightLanes()) {
                opposite_lane = getInstance.FindLaneByUuid(right_uuid);
            }
        }

        //closest current lane point
        std::pair<freicar::mapobjects::Point3D, int> current_closest_point = current_lane->GetClosestLanePoint(current_position);

        Eigen::Vector3f last_point = published_path_temp.at(published_path_temp.size() -1);
        freicar::mapobjects::Point3D current_last_lane_point(last_point.x(), last_point.y(), last_point.z());
        std::pair<freicar::mapobjects::Point3D, int> opp_last_point = opposite_lane->GetClosestLanePoint(current_last_lane_point);
        std::pair<freicar::mapobjects::Point3D, int> opp_closest_point = opposite_lane->GetClosestLanePoint(current_closest_point.first);

        //std::pair<freicar::mapobjects::Point3D, int> curr_end_point = current_lane->GetClosestLanePoint(opp_closest_point.first);
        //start point index to change the path points to oivertake
        int start_point = 0;
        for(size_t i = 0; i < published_path_temp.size(); ++i) {
            float d = sqrt(pow(published_path_temp.at(i).x() - current_closest_point.first.x(), 2) + pow(published_path_temp.at(i).y()  - current_closest_point.first.y(), 2));

            if (d <  0.1){
                std::cout << "Found start point ...................."  <<std::endl;
                start_point = i;
                break;
            }
        }

        //end point intex untill where we have to change the path points
        int end_point = 0;
        for(size_t i = published_path_temp.size() -1; i >= 0; i--) {
            float d = sqrt(pow(published_path_temp.at(i).x() - current_closest_point.first.x(), 2) + pow(published_path_temp.at(i).y()  - current_closest_point.first.y(), 2));

            if (d >  3 && d < 3.2){
                std::cout << "Found end point ...................."  <<std::endl;
                end_point = i;
                break;
            }
        }

        std::cout<<current_position.x()<< " "<<current_position.y() <<std::endl;
        std::cout<<last_point.x()<< " "<<last_point.y()<<std::endl;
        std::cout<<opp_last_point.first.x()<< " "<<opp_last_point.first.y() <<std::endl;
        std::cout<<opp_closest_point.first.x()<< " "<<opp_closest_point.first.y() <<std::endl;

        //Get new plan in opposite lane
        auto plan = freicar::planning::lane_follower::GetPlan(opp_last_point.first, command, 10, 80);
        std::cout<<"New overtake published path ... "<<plan.steps.size()<<std::endl;

        current_overtake_path.clear();
        //Initialize full overtake path
        for(size_t i = 0; i < plan.steps.size(); ++i) {
            Eigen::Vector3f path_point(plan.steps[i].position.x(), plan.steps[i].position.y(), 0.0);
            current_overtake_path.push_back(path_point);

        }

        std::cout<<"Start and end index: "<<start_point << " " << end_point<<std::endl;
        //Replace path from start to end index in current lane points with opposite lane points
        for(size_t i = start_point; i <=end_point; i++) {
            float min_dis = 99999;
            int min_index;
            for(size_t j = 0; j < current_overtake_path.size(); j++) {
                float d = sqrt(pow(current_overtake_path.at(j).x() - published_path_temp.at(i).x(), 2) + pow(current_overtake_path.at(j).y() - published_path_temp.at(i).y(), 2));
                if(d < min_dis){
                    min_dis = d;
                    min_index = j;
                }
            }
            std::cout<<current_overtake_path.at(min_index).x()<< " " << current_overtake_path.at(min_index).y()<<std::endl;
            published_path_temp.at(i) = current_overtake_path.at(min_index);
        }

        rais_control_msg.path_segment.header.stamp = ros::Time::now();
        rais_control_msg.path_segment.header.frame_id = "map";
        rais_control_msg.path_segment.header.seq = 0;


        for(size_t i = 0; i < published_path_temp.size(); i++) {
            pose_msg.pose.position.x = published_path_temp.at(i).x();
            pose_msg.pose.position.y = published_path_temp.at(i).y();
            pose_msg.pose.position.z = published_path_temp.at(i).z();

            //std::cout<<pose_msg.pose.position.x<< " " << pose_msg.pose.position.y<<std::endl;

            //pose_msg.pose.orientation.w = plan.steps[i].path_description;
            //pose_msg.pose.orientation.w = getDirectionValues(plan.steps[i].path_description);
            //std::cout << i << " " << plan.steps[i].position.x()<< " "<< plan.steps[i].position.y()<< " " << plan.steps[i].path_description << " " << pose_msg.pose.orientation.w<<std::endl;
            rais_control_msg.path_segment.poses.push_back(pose_msg);


        }
        Overtake_PlanPublish(published_path_temp, 1.0, 0.0, 0.0, 300, "plan", broadcast);

        path_segment.publish(rais_control_msg);

        obstacleDistance.data = 100;
        std::cout << "Sending obstacle again more than 100 after publishing overtake.............. " << obstacleDistance.data <<  std::endl;
        obstacle_status.publish(obstacleDistance);
        send_obstacle = false;

        std::cout << "Path Published ....." << command <<std::endl;
        ROS_INFO("PATH published...................");
        //reset the command to straight
        command = freicar::enums::PlannerCommand::STRAIGHT;

    }

}

/*
 * Call back function to recieive goal status
 *
 */
void planner::GoalReachedStatusReceived(const std_msgs::Bool goal_message) {
    goal_reached = goal_message.data;
    command_changed = false;
    if(goal_reached == true){
        PublishNewPlan();
    }
    goal_reached = false;
    std::cout << "Goal Reached ..... "<<goal_reached << std::endl;
}

/*
 * Call back function to recieive commands
 *
 */
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

/*
 * This function publishes new path on request based on current location of agent
 *
 */
void planner::PublishNewPlan() {
    geometry_msgs::PoseStamped pose_msg;
    raiscar_msgs::ControllerPath rais_control_msg;

    auto plan = freicar::planning::lane_follower::GetPlan(current_position, command, 10, 80);

    PublishPlan(plan, 0.0, 1.0, 0.0, 300, "plan", broadcast);

    rais_control_msg.path_segment.header.stamp = ros::Time::now();
    rais_control_msg.path_segment.header.frame_id = "map";
    rais_control_msg.path_segment.header.seq = 0;

    published_path.clear();
    std::cout<<"clearing published path ... "<<published_path.size()<<std::endl;

    for(size_t i = 0; i < plan.steps.size(); ++i) {
        pose_msg.pose.position.x = plan.steps[i].position.x();
        pose_msg.pose.position.y = plan.steps[i].position.y();
        pose_msg.pose.position.z = plan.steps[i].position.z();

        Eigen::Vector3f path_point(plan.steps[i].position.x(), plan.steps[i].position.y(), 0.0);
        published_path.push_back(path_point);

        //pose_msg.pose.orientation.w = plan.steps[i].path_description;
        pose_msg.pose.orientation.w = getDirectionValues(plan.steps[i].path_description);
        //std::cout << i << " " << plan.steps[i].position.x()<< " "<< plan.steps[i].position.y()<< " " << plan.steps[i].path_description << " " << pose_msg.pose.orientation.w<<std::endl;
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


void planner::InitializeBestParticle2(const freicar_common::FreiCarAgentLocalization msg){
    if(msg.name == agent_name){
        freicar::mapobjects::Point3D current_point(msg.current_pose.transform.translation.x, msg.current_pose.transform.translation.y, msg.current_pose.transform.translation.z);
        current_position = current_point;
        //std::cout << "Car Local: " <<  msg.current_pose.transform.translation.x <<", "<< msg.current_pose.transform.translation.y << std::endl;

        auto& map_instance = freicar::map::Map::GetInstance();
        closest_lane_uuid = map_instance.FindClosestLanePoints(current_point.x(), current_point.y(), current_point.z(), 1)[0].first.GetLaneUuid();
        const freicar::mapobjects::Lane *current_lane = map_instance.FindLaneByUuid(closest_lane_uuid);
        const freicar::mapobjects::Stopline* stop_line = current_lane->GetStopLine();
        std::vector<const freicar::mapobjects::Roadsign*>  roadSigns = current_lane->GetRoadSigns();
        bool hasStopLine = current_lane->HasRoadSign("Stop");
        bool hasRightOfWay = current_lane->HasRoadSign("RightOfWay");
        int a=2;

        //std::cout << "Localization11: " <<  msg.poses.data()->position.x <<", "<< msg.poses.data()->position.y << std::endl;
        if(hasStopLine){
            float distance = current_lane->GetStopLine()->GetPosition().ComputeDistance(current_point);
            if(distance >= 0.4 && distance <= 1.2 && (ros::Time::now() - last_stop_publish_time).toSec() > 6.0){
                stopDistance.data = static_cast<float>(distance);
                std::cout << "Sending again.............. " << (ros::Time::now() - last_stop_publish_time).toSec() << "            "<< distance <<       std::endl;
                stop_status.publish(stopDistance);
                if(distance < 0.6 ){
                    last_stop_publish_time = ros::Time::now();
                    stopDistance.data = 100.0;

                    stop_status.publish(stopDistance);
                    stop_status.publish(stopDistance);
                    stop_status.publish(stopDistance);
                    std::cout << "updatae time.............. " << (ros::Time::now() - last_stop_publish_time).toSec()  << std::endl;
                    //stopDistance.data = static_cast<float>(distance);
                    //stop_status.publish(stopDistance);
                }
            }
        }

        //implement right of way
        if(hasRightOfWay ){
            float rightOfWayDistance;
            for(int i= 0; i < roadSigns.size(); i++){
                if(roadSigns.at(i)->GetSignType() == "RightOfWay"){
                    rightOfWayDistance = roadSigns.at(i)->GetPosition().ComputeDistance(current_point);
                    std::cout<<"right of way distance: "<<rightOfWayDistance<<std::endl;

                    if(rightOfWayDistance < 1.0 ){
                        if(current_boundig_bxs.layout.dim.size() > 0 && current_boundig_bxs.layout.dim[0].size > 0){
                            std::cout<<"Bounding boxes: "<< current_boundig_bxs.layout.dim[0].size<<std::endl;
                            for(auto j=0; j < current_boundig_bxs.layout.dim[0].size; j++){
                                float x = current_boundig_bxs.data[j*6 + 4];
                                float y = current_boundig_bxs.data[j*6 + 5];
                                float radius = 2.5;
                                std::cout<<"Objects at  junction distance "<< (pow(x, 2) + pow(y , 2) )<<std::endl;
                                if ((pow(x, 2) + pow(y , 2) )< pow(radius, 2)){
                                    stopDistance.data = static_cast<float>(rightOfWayDistance);
                                    stop_status.publish(stopDistance);
                                    std::cout<<"Objects in Junction wait ....... "<< x << " " << y<<std::endl;
                                    std::cout<<" Sending rightOfWay stop distance with objects at junction : "<<stopDistance<<std::endl;
                                }else{
                                    std::cout<<"Publishing distance from Righ of way when objects are far ... "<< 100.0<<std::endl;
                                    stopDistance.data = static_cast<float>(100.0);
                                    stop_status.publish(stopDistance);
                                }

                            }
                        }else{
                            std::cout<<"Publishing distance from Right of way with no bounding boxes and right of way less than 1... "<< 100.0<<std::endl;
                            stopDistance.data = static_cast<float>(100.0);
                            stop_status.publish(stopDistance);
                        }

                    }else{
                        std::cout<<"Publishing distance from Righ of way with no bounding boxes... "<< 100.0<<std::endl;
                        stopDistance.data = static_cast<float>(100.0);
                        stop_status.publish(stopDistance);
                    }

                }


            }
        }

    }

}

/*
 * Initializes current location of the agent and also implements logic for stop and rightOfWay Signals
 */
void planner::InitializeBestParticle(const geometry_msgs::PoseArray msg){
    freicar::mapobjects::Point3D current_point(msg.poses.data()->position.x, msg.poses.data()->position.y, msg.poses.data()->position.z);
    current_position = current_point;
    //std::cout<< "Best particle: "<<std::endl;


    double x_ang = msg.poses.data()->orientation.x;
    double y_ang = msg.poses.data()->orientation.y;
    double z_ang = msg.poses.data()->orientation.z;
    double w_ang = msg.poses.data()->orientation.w;
    current_angle = atan2(2.0f * (w_ang * z_ang + x_ang * y_ang), 1.0f - 2.0f * (y_ang * y_ang + z_ang * z_ang));
    ////std::cout<< "Best particle: "<<std::endl;

    auto& map_instance = freicar::map::Map::GetInstance();
    closest_lane_uuid = map_instance.FindClosestLanePoints(current_point.x(), current_point.y(), current_point.z(), 1)[0].first.GetLaneUuid();
    const freicar::mapobjects::Lane *current_lane = map_instance.FindLaneByUuid(closest_lane_uuid);
    const freicar::mapobjects::Stopline* stop_line = current_lane->GetStopLine();
    std::vector<const freicar::mapobjects::Roadsign*>  roadSigns = current_lane->GetRoadSigns();
    bool hasStopLine = current_lane->HasRoadSign("Stop");
    bool hasRightOfWay = current_lane->HasRoadSign("RightOfWay");
    int a=2;

    //std::cout << "Localization11: " <<  msg.poses.data()->position.x <<", "<< msg.poses.data()->position.y << std::endl;
    if(hasStopLine){
        float distance = current_lane->GetStopLine()->GetPosition().ComputeDistance(current_point);
        if(distance >= 0.4 && distance <= 1.2 && (ros::Time::now() - last_stop_publish_time).toSec() > 6.0){
            stopDistance.data = static_cast<float>(distance);
            std::cout << "Sending again.............. " << (ros::Time::now() - last_stop_publish_time).toSec() << "            "<< distance <<       std::endl;
            stop_status.publish(stopDistance);
            if(distance < 0.6 ){
                last_stop_publish_time = ros::Time::now();
                stopDistance.data = 100.0;

                stop_status.publish(stopDistance);
                stop_status.publish(stopDistance);
                stop_status.publish(stopDistance);
                std::cout << "updatae time.............. " << (ros::Time::now() - last_stop_publish_time).toSec()  << std::endl;
                //stopDistance.data = static_cast<float>(distance);
                //stop_status.publish(stopDistance);
            }
        }
    }

    //implement right of way
    if(hasRightOfWay ){
        float rightOfWayDistance;
        for(int i= 0; i < roadSigns.size(); i++){
            if(roadSigns.at(i)->GetSignType() == "RightOfWay"){
                rightOfWayDistance = roadSigns.at(i)->GetPosition().ComputeDistance(current_point);
                std::cout<<"right of way distance: "<<rightOfWayDistance<<std::endl;

                if(rightOfWayDistance < 1.0 ){
                    if(current_boundig_bxs.layout.dim.size() > 0 && current_boundig_bxs.layout.dim[0].size > 0){
                        std::cout<<"Bounding boxes: "<< current_boundig_bxs.layout.dim[0].size<<std::endl;
                        for(auto j=0; j < current_boundig_bxs.layout.dim[0].size; j++){
                            float x = current_boundig_bxs.data[j*6 + 4];
                            float y = current_boundig_bxs.data[j*6 + 5];
                            float radius = 2.5;
                            std::cout<<"Objects at  junction distance "<< (pow(x, 2) + pow(y , 2) )<<std::endl;
                            if ((pow(x, 2) + pow(y , 2) )< pow(radius, 2)){
                                stopDistance.data = static_cast<float>(rightOfWayDistance);
                                stop_status.publish(stopDistance);
                                std::cout<<"Objects in Junction wait ....... "<< x << " " << y<<std::endl;
                                std::cout<<" Sending rightOfWay stop distance with objects at junction : "<<stopDistance<<std::endl;
                            }else{
                                std::cout<<"Publishing distance from Righ of way when objects are far ... "<< 100.0<<std::endl;
                                stopDistance.data = static_cast<float>(100.0);
                                stop_status.publish(stopDistance);
                            }

                        }
                    }else{
                        std::cout<<"Publishing distance from Right of way with no bounding boxes and right of way less than 1... "<< 100.0<<std::endl;
                        stopDistance.data = static_cast<float>(100.0);
                        stop_status.publish(stopDistance);
                    }

                }else{
                    std::cout<<"Publishing distance from Righ of way with no bounding boxes... "<< 100.0<<std::endl;
                    stopDistance.data = static_cast<float>(100.0);
                    stop_status.publish(stopDistance);
                }

            }


        }
    }

}

/*
 * Logic to identify collision with other objects
 */

void planner::BoundingBoxRecieved(const std_msgs::Float32MultiArray msg) {
    float min_distance=100.0, max_distance;

    current_boundig_bxs = msg;

    std::cout<<"bounding box recieved .."<<std::endl;

    for(auto i=0; i < msg.layout.dim[0].size;i++){
        float top_x = msg.data[i*6 + 0];
        float top_y = msg.data[i*6 + 1] ;
        float bottom_x= msg.data[i*6 + 2];
        float bottom_y = msg.data[i*6 + 3];
        float point_x = msg.data[i*6 + 4];
        float point_y = msg.data[i*6 + 5];

        float width = bottom_x-top_x;
        float height = bottom_y-top_y;
        float x_center = top_x + width/2;
        float y_center = top_y + height/2;

        geometry_msgs::TransformStamped tf_msg, ra_t_map_msg, fa_t_map_msg;
        geometry_msgs::TransformStamped front_axis_tf_msg;
        tf2::Stamped <tf2::Transform> map_t_ra;
        tf2::Stamped <tf2::Transform> ra_t_map;
        tf2::Stamped <tf2::Transform> fa_t_map;

        geometry_msgs::TransformStamped cam_t_map_msg;
        tf2::Stamped <tf2::Transform> cam_t_map;
        try {
            //converting camera frame to map
            cam_t_map_msg = tf_buffer_.lookupTransform("map", agent_name+"/zed_camera", ros::Time(0));

        } catch (tf2::TransformException &ex) {
            ROS_WARN_STREAM(ex.what());
        }
        tf2::convert(cam_t_map_msg, cam_t_map);

        Eigen::Transform<float, 3, Eigen::Affine> t = Eigen::Transform<float, 3, Eigen::Affine>::Identity();
        t.translate(Eigen::Vector3f(point_x, point_y, 0.0f));
        std::cout << cam_t_map.getOrigin().x()<< " " << cam_t_map.getOrigin().y()<<std::endl;
        t.translate(Eigen::Vector3f(cam_t_map.getOrigin().x(), cam_t_map.getOrigin().y(), 0.));
        t.rotate(Eigen::Quaternionf(cam_t_map.getRotation().getW(), cam_t_map.getRotation().getX(),
                                    cam_t_map.getRotation().getY(), cam_t_map.getRotation().getZ()));

        //publish to rviz
        visualization_msgs::Marker marker;
        marker.header.frame_id = "/map";
        marker.header.stamp = ros::Time::now();
        marker.ns = "bbx";
        marker.id = 0;
        marker.type = visualization_msgs::Marker::SPHERE;;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = t.translation().x();
        marker.pose.position.y = t.translation().y();
        marker.pose.position.z = 0;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;
        // Set the scale of the marker -- 1x1x1 here means 1m on a side
        marker.scale.x = 1.0;
        marker.scale.y = 1.0;
        marker.scale.z = 1.0;
        marker.color.r = 0.0f;
        marker.color.g = 0.0f;
        marker.color.b = 1.0f;
        marker.color.a = 1.0;
        marker_pub.publish(marker);

        float x_obj = point_x, y_obj = point_y;
        //std::cout <<x_obj<<" "<<y_obj<<"--> "<< "map frame: "<<t.translation().x() << " " << t.translation().y()<< "--> "<<current_position.x()<<" "<<current_position.y() <<std::endl;

        Eigen::Vector3f obj_point(t.translation().x(), t.translation().y(), 0.0);
        float distance = check_collision(obj_point, point_x, point_y);
        if(distance < min_distance){
            min_distance = distance;
        }

    }


    if(min_distance < 100 && send_obstacle){
        last_stop_publish_time = ros::Time::now();
        obstacleDistance.data = min_distance;
        std::cout << "Sending obstacle again.............. " << obstacleDistance.data <<  std::endl;
        obstacle_status.publish(obstacleDistance);
    }else if((ros::Time::now() - last_stop_publish_time).toSec() > 2.0){
        send_obstacle = true;
        obstacleDistance.data = 100;
        std::cout << "Sending obstacle again more than 100.............. " << obstacleDistance.data <<  std::endl;

        obstacle_status.publish(obstacleDistance);
    }




}


float planner::check_collision(Eigen::Vector3f obj_point, float px, float py){
    float collision_distance = 100.0f;

    std::cout<< "Obj detected .. "<<published_path.size()<<": "<< obj_point.x() << " " << obj_point.y() << " cam: "<<px<<" "<<py<<std::endl;
    for(int i = 0; i < published_path.size(); i++){
        //std::cout<< "entering .. "<<i<<std::endl;
        float x = obj_point.x();
        float y = obj_point.y();
        float centre_x, centre_y;
        if(i < published_path.size()){
            centre_x = published_path.at(i).x();
        }else{
            break;
        }
        if(i < published_path.size()){
            centre_y = published_path.at(i).y();
        }else{
            break;
        }

        float radius = obstacle_radius;
        //std::cout<< "comparing .. "<<i << " :"<< published_path.at(i).x()<<" "<< published_path.at(i).y()<< ": "<<pow((x - centre_x), 2) + pow((y - centre_y), 2)<<std::endl;
        if((pow((x - centre_x), 2) + pow((y - centre_y), 2) )< pow(radius, 2)){
            //collision_distance = sqrt(pow((current_position.x()- centre_x), 2) + pow((current_position.y()- centre_y), 2));
            collision_distance = sqrt(pow(px, 2) + pow(py, 2));
            std::cout<<"collision in path.....  "<<collision_distance <<"           : with circle "<< pow((x - centre_x), 2) + pow((y - centre_y), 2) <<std::endl;
            return collision_distance;
        }
    }
    //std::cout<<"returning from compare.....  "<<collision_distance<<std::endl;
    return collision_distance;
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
