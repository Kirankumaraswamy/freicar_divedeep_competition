/*
 * Author: Johan Vertens (vertensj@informatik.uni-freiburg.de)
 * Project: FreiCAR
 * Do NOT distribute this code to anyone outside the FreiCAR project
 */
/* A ROS implementation of the Pure pursuit path tracking algorithm (Coulter 1992).
   Terminology (mostly :) follows:
   Coulter, Implementation of the pure pursuit algoritm, 1992 and
   Sorniotti et al. Path tracking for Automated Driving, 2017.
 */
#include <string>
#include <cmath>
#include <algorithm>
#include <fstream>
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
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <kdl/frames.hpp>
#include <raiscar_msgs/ControlReport.h>
#include "raiscar_msgs/ControlCommand.h"
#include "std_msgs/Bool.h"
#include "controller.h"
#include <filesystem>
#include "geometry_msgs/PoseArray.h"
using std::string;
class PurePursuit: public controller
{
public:
    //! Constructor
    PurePursuit();
    //! Run the controller.
    void run();
    std::vector<double> cross_point_error;
    std::vector<double> heading_error;
private:
    void controller_step1(geometry_msgs::PoseArray msg);
    void controller_step(nav_msgs::Odometry odom);
    double ld_dist_;
};
PurePursuit::PurePursuit()
{
    // Get parameters from the parameter server
    nh_private_.param<double>("lookahead_dist", ld_dist_, 0.7);
    std::cout << "*********Pure Pursuit controller started...***********" << std::endl;
}
/*
 * Implement your controller here! The function gets called each time a new odometry is incoming.
 * The path to follow is saved in the variable "path_". Once you calculated the new control outputs you can send it with
 * the pub_acker_ publisher.
 */
void PurePursuit::controller_step(nav_msgs::Odometry odom) {
    // Code blocks that could be useful:
    // The following code block could receive the current pose (saved in map_t_ra)
    geometry_msgs::TransformStamped tf_msg, ra_t_map_msg, fa_t_map_msg;
    geometry_msgs::TransformStamped front_axis_tf_msg;
    tf2::Stamped <tf2::Transform> map_t_ra;
    tf2::Stamped <tf2::Transform> ra_t_map;
    tf2::Stamped <tf2::Transform> fa_t_map;


    try {
        //converting map frame to rear axis
        tf_msg = tf_buffer_.lookupTransform(rear_axis_frame_id_, map_frame_id_, ros::Time(0));
        //converting rear axis to map frame
        ra_t_map_msg = tf_buffer_.lookupTransform(map_frame_id_, rear_axis_frame_id_, ros::Time(0));

        //converting front axis to map frame
        fa_t_map_msg = tf_buffer_.lookupTransform(map_frame_id_, front_axis_frame_id_, ros::Time(0));
    } catch (tf2::TransformException &ex) {
        ROS_WARN_STREAM(ex.what());
    }
    tf2::convert(tf_msg, map_t_ra);
    tf2::convert(ra_t_map_msg, ra_t_map);
    tf2::convert(fa_t_map_msg, fa_t_map);

    double steering_angle = 0;
    double vel = 0;
    bool brake = false;
    double x_ld, y_ld;

    float car_x = (fa_t_map.getOrigin().x() + ra_t_map.getOrigin().x())/2;
    float car_y = (fa_t_map.getOrigin().y() + ra_t_map.getOrigin().y())/2;

    //Calculating car orientation using w.r.t map frame using rear axis and odometry(center of the car) w.r.t map frame
    double car_angle = atan2((fa_t_map.getOrigin().y() - ra_t_map.getOrigin().y()),
                             (fa_t_map.getOrigin().x() - ra_t_map.getOrigin().x())) * (180 / 3.142);

    //if (!path_.size() > 0) {
        //completion_advertised_ = false;
    //}
    if (path_.size() > 0 && !goal_reached_) {

        tf2::Transform t1, t2;
        double d1, d2;
        double x1, y1, z1;
        double x2, y2, z2;
        double minimum_dist = std::numeric_limits<double>::infinity();
        double cross_track_error_dist;
        double path_angle;
        for (int i = 1; i < path_.size(); i++) {
            // Converting values of path points to rear axis frame
            t1 = map_t_ra * path_.at(i - 1);
            x1 = t1.getOrigin().x();
            y1 = t1.getOrigin().y();
            z1 = t1.getOrigin().z();
            t2 = map_t_ra * path_.at(i);
            x2 = t2.getOrigin().x();
            y2 = t2.getOrigin().y();
            z2 = t2.getOrigin().z();

            //find distance of the two neihboring points w.r.t to the rear axis frame
            d1 = sqrt(pow(x1, 2) + pow(y1, 2) + pow(z1, 2));
            d2 = sqrt(pow(x2, 2) + pow(y2, 2) + pow(z2, 2));

            //These distance are used to cross track error
            double odom_dist1 = sqrt(pow(path_.at(i - 1).getOrigin().x() - car_x, 2) +
                                     pow(path_.at(i - 1).getOrigin().y() - car_y, 2));
            double odom_dist2 = sqrt(pow(path_.at(i).getOrigin().x() - car_x, 2) +
                                     pow(path_.at(i).getOrigin().y() - car_y, 2));

            //Calculating slope of 2 points in rear axis frame
            double m = (y2 - y1) / (x2 - x1);

            if (minimum_dist > (odom_dist1 + odom_dist2) / 2) {
                minimum_dist = (odom_dist1 + odom_dist2) / 2;
                //slope in map frame
                double map_m = (path_.at(i).getOrigin().y() - path_.at(i - 1).getOrigin().y()) /
                               (path_.at(i).getOrigin().x() - path_.at(i - 1).getOrigin().x());
                //Logic to find cross point error.
                double A, B, C;
                A = map_m;
                B = -1;
                C = path_.at(i - 1).getOrigin().y() - map_m * path_.at(i - 1).getOrigin().x();
                //perpendicular distance between the point(middle of car axis) and the line(formed by 2 path points)
                cross_track_error_dist =
                        abs(A * ra_t_map.getOrigin().x() + B * ra_t_map.getOrigin().y() + C) / sqrt(A * A + B * B);
                ////std::cout<<"index : " << i <<" d: " << minimum_dist << " error: " << cross_track_error_dist<<std::endl;
                path_angle = atan(map_m) * (180 / 3.142);
                ////std::cout<<" index:" << i << std::endl;
            }

            //select the two neighboring points where look ahead point lie between them.
            //by finding the point of intersection of circle and straigh line between the two points
            if (ld_dist_ >= d1 && ld_dist_ <= d2 && !goal_reached_) {
                double k_end = m; // Slope of line defined by the last path pose
                double l_end = y1 - k_end * x1;
                double a = 1 + k_end * k_end;
                double b = 2 * m * l_end;
                double c = l_end * l_end - ld_dist_ * ld_dist_;
                double D = sqrt(b * b - 4 * a * c);
                x_ld = (-b + copysign(D, vmax_)) / (2 * a);
                y_ld = k_end * x_ld + l_end;

                //set the steering angle corresponding to look ahead point w.r.t to rear axis frame
                steering_angle =  std::min( atan2(2 * y_ld * L_, ld_dist_ * ld_dist_)  + m * steering_penalty, delta_max_ );
                //steering_angle =  std::min( atan2(2 * y_ld * L_, ld_dist_ * ld_dist_), delta_max_ );



                if(obstacle_distance <= 2.5){
                    if(obstacle_distance < 1.75){
                        vel = 0;
                        cmd_control_.steering = steering_angle;
                        cmd_control_.throttle = 0.0;
                        cmd_control_.throttle_mode = 0;
                        cmd_control_.brake = true;
                        pub_acker_.publish(cmd_control_);
                        sleep(0.5);
                        if(standing_count > 5){
                            sendOvertakeStatus(true);
                        }
                    }else{
                        //vel = des_v_;
                        vel = des_v_ - (des_v_ * exp(- abs((obstacle_distance - 1.5))));
                    }
                }else if(stop_distance > 0.3 && stop_distance <= 1.2) {
                    //stop for 3 seconds near stop signal
                    //vel = des_v_;
                    if (stop_distance <= 0.6) {
                        cmd_control_.steering = 0;
                        cmd_control_.throttle = 0.0;
                        cmd_control_.throttle_mode = 0;
                        cmd_control_.brake = true;
                        pub_acker_.publish(cmd_control_);
                        sleep(1);
                        stop_distance = 100.0;
                    }else{
                        //vel = des_v_/2;
                        vel = des_v_ - (des_v_ * exp(-abs((stop_distance - 0.6))));
                    }
                }

                else {
                    //hard coded velocity value
                    //vel = des_v_ - (des_v_/3) * exp(-abs(m));
                    vel = des_v_;
                }

                break;
            }
            //if we can't find the best two neigboring path points, it means we have almost approched goal.
            if (i >= path_.size() - 1) {
                std::cout << "Path Point Size:" << path_.size() << std::endl;
                goal_reached_ = true;
                completion_advertised_ = false;
            }
        }
        if (!goal_reached_) {
            //add the cross point and heading errors in this time interval.
            cross_point_error.push_back(cross_track_error_dist);
            heading_error.push_back(abs(abs(car_angle) - abs(path_angle)));
            ////std::cout<< " car angle : " << car_angle<<" path angle : " << path_angle<<" distance :" << minimum_dist << std::endl;
        }
    }
    //path completion check. To make sure we are close to the goal point by using position tollerance value.
    //if we haven't reached the goal then slowly follow the previous steering angle.
    if(path_.size()>0 && goal_reached_){
        tf2::Transform t3;
        double d3;
        double x3, y3, z3;
        //our goal is to make last path point to stay closer to the rear axis of the car.
        t3 = map_t_ra * path_.at(path_.size() - 1) ;
        x3 = t3.getOrigin().x();
        y3 = t3.getOrigin().y();
        z3 = t3.getOrigin().z();
        d3 = sqrt(pow(x3, 2) + pow(y3, 2));
        if (fabs(d3) <= pos_tol_) {
            vel = 0;
            brake = true;
            //once we reach the goal, save the error values to plot the graph using plot.py file.
            /*std::ofstream myfile ("/home/freicar/freicar_ws/src/freicar_ss21_exercises/01-01-control-exercise/freicar_control/src/controller/plot.txt");
            if (myfile.is_open())
            {
                myfile << "CROSS_POINT_ERROR: ";
                for(int count = 0; count < cross_point_error.size(); count ++){
                    myfile << cross_point_error[count] << " " ;
                }
                myfile << "\n";
                myfile << "HEADING_ERROR: ";
                for(int count = 0; count < cross_point_error.size(); count ++){
                    myfile << heading_error[count] << " " ;
                }
                myfile << "\n";
                myfile.close();
            }
            else std::cout << "Unable to open file";*/
            //std::cout << "value :" << d3 << "X:" << x3 << "y3:" << y3 <<std::endl;
        } else {
            steering_angle = cmd_control_.steering;
            vel = 0.05;
        }
    }

    // The following code block sends out the boolean true to signal that the last waypoint is reached:
    sendGoalMsg(goal_reached_);
    //The following code block could be used for sending out a tf-pose for debugging
    target_p_.transform.translation.x = x_ld; // dummy value
    target_p_.transform.translation.y = y_ld; // dummy value
    target_p_.transform.translation.z = 0; // dummy value
    target_p_.header.frame_id = map_frame_id_;
    target_p_.header.stamp = odom.header.stamp;
    target_p_.header.stamp = ros::Time::now();
    tf_broadcaster_.sendTransform(target_p_);

    // The following code block can be used to control a certain velocity using PID control
    //double pid_vel_out = 0.0;
//else {
    //    pid_vel_out = vel;
    //    vel_pid.resetIntegral();
   // }

   //std::cout << "Velocity1:" << vel << " Steering1:" << steering_angle << std::endl;
    // The following code block can be used to send control commands to the car
    //setting the boundries of streeing angle
    //steering_angle = std::min(1.0, std::max(-1.0, steering_angle));
    cmd_control_.steering = steering_angle;
    cmd_control_.throttle =  vel;
    cmd_control_.throttle_mode = 0;
    cmd_control_.brake = brake;
    //cmd_control_.throttle = std::min(cmd_control_.throttle, throttle_limit_);
    //cmd_control_.throttle = std::max(std::min((double)cmd_control_.throttle, 1.0), 0.0);
    pub_acker_.publish(cmd_control_);
}

void PurePursuit::run()
{
    ros::spin();
}
int main(int argc, char**argv)
{
    ros::init(argc, argv, "pure_pursuit_controller");
    PurePursuit controller;
    controller.run();

    return 0;
}
