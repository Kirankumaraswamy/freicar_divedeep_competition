/*
 * Author: Johan Vertens (vertensj@informatik.uni-freiburg.de)
 * Project: FreiCAR
 * Do NOT distribute this code to anyone outside the FreiCAR project
 */

#include <map_core/freicar_map.h>
#include "particle_filter.h"
#include <tf2/transform_datatypes.h>
#include <tf2/convert.h>

const unsigned int sizeOfParticles = 5000;

particle_filter::particle_filter(freicar::map::Map* map, std::shared_ptr<ros_vis> vis, bool use_lane_reg, std::string agent_name_param): visualizer_(vis)
{
    map_ = map;
    particles_init_ = false;
    odo_init_ = false;
    map_init_ = false;
    map_init_ = this->InitMap(map);
    latest_x_vel = 0.0f;
    particle_memory_.resize(BEST_PARTICLE_HISTORY);
    memory_init_ = false;
    memory_insert_cnt_ = 0;

    agent_name = agent_name_param;

    // Spread particles over whole map
    if(!particles_init_){
        InitParticles();
        visualizer_->SendPoses(particles_, "particles", "world");
        particles_init_ = true;
    }
    use_lane_reg_ = use_lane_reg;
    // Important: Map has to be initialized before calling this !
    sensor_model_ = std::make_shared<sensor_model>(kd_data_, sign_data_, visualizer_, use_lane_reg_);
}

float particle_filter::getSumWeights(){
    float sum = 0.0f;
    for(size_t i = 0; i < particles_.size(); i++){
        sum += particles_.at(i).weight;
    }
    return sum;
}

/*
 * ##############################################IMPLEMENT ME###########################################################
 * Implement Roulette Sampling here. The particles are stored in "particles_" (member variable)
 * Hint: Create a new set of particles std::vector<Particle> and then at the end
 * set particles_ = new_particles;
 */
void particle_filter::RouletteSampling(){
    std::vector<Particle> new_particles;
    float sum =0;
    float bucket[particles_.size()];
    for(int i=0;i<particles_.size();i++){
        sum += particles_[i].weight;
        bucket[i] = sum;
    }

    for(int i=0;i<particles_.size();i++){
        float rand_val =  static_cast <float> (rand()) /( static_cast <float> (RAND_MAX));
        int particle_index = binarySearch(bucket, 0,particles_.size()-1, rand_val);
        new_particles.push_back(particles_[particle_index]);
    }
    particles_ = new_particles;
}

int particle_filter::binarySearch(float *arr, int l, int r, int x)
{
    while (r > l) {
        int mid = l + (r - l) / 2;

        // If the element is present at the middle
        // itself
        if (arr[mid] == x)
            return mid;

        // If element is smaller than mid, then
        // it can only be present in left subarray
        if (arr[mid] > x)
            r= mid -1 ;

        // Else the element can only be present
        // in right subarray
        l = mid +1;
    }

    // We reach here when element is not
    // present in array
    return r;
}


/*
 * ##############################################IMPLEMENT ME###########################################################
 * Implement Low Variance Sampling here. The particles are stored in "particles_" (member variable)
 * Hint: Create a new set of particles std::vector<Particle> and then at the end
 * set particles_ = new_particles;
 * If you need uniform distributions take a look on: std::uniform_real_distribution
 */
void particle_filter::LowVarianceSampling(){
    float normalizer = 0.0;
    for (int i = 0; i < particles_.size(); i++){
        normalizer = normalizer + particles_[i].weight;
    }

    std::vector<Particle> new_particles;
    std::vector<int> new_idx;
    int M = particles_.size();
    std::uniform_real_distribution<float> u_dist(0, 1.0/float(M));
    float U =u_dist(generator_);
    float c = particles_[0].weight / normalizer;
    int i = 0;
    for(int m=1 ; m<M ; m++){

        while(U > c){
            i++;
            c += particles_[i].weight /normalizer ;
        }
        U = U + 1.0/float(M);
        Particle p = particles_[i];
        new_particles.push_back(p);
    }
    particles_ = new_particles;
}


/*
 * #############################IMPLEMENT ME############################################################################
 * Implement your particle initialization here. The variable maxes_mins holds the maximum and minimum values of x and y
 * of the map. Spawn random particles inside this rectangle. The particles pose is represented
 * as Eigen Transform (Eigen::Transform<float,3,Eigen::Affine>). The particles should be stored in "particles_" (member).
 * The class Particle is defined in data_types.hpp
 */
void particle_filter::InitParticles(){
    memory_init_ = false;
    memory_insert_cnt_ = 0;

    if(!particles_.empty()){
        std::cout << "Particles are not empty... clearing..." << std::endl;
        particles_.clear();
    }

    Eigen::Vector4f maxes_mins = map_helper::getMaxesMap(map_);
    std::cout << "Map extrema: Min_x: " << maxes_mins[0] << " Max_x: " << maxes_mins[1] << " Min_y: " << maxes_mins[2] << " Max_y: " << maxes_mins[3] << std::endl;

    //Keep above code, and implement under this block! /////////////////////////////////////////////////////////////////

    //std::uniform_real_distribution<float> generate_x(maxes_mins[0], maxes_mins[1]);
    //std::uniform_real_distribution<float> generate_y(maxes_mins[2], maxes_mins[3]);
    //std::uniform_real_distribution<float> generate_rot(-M_PI, M_PI);_
    float init_x, init_y, heading;

    ros::param::get("/freicar_"+agent_name+"_carla_proxy/spawn/x", init_x);
    ros::param::get("/freicar_"+agent_name+"_carla_proxy/spawn/y", init_y);
    ros::param::get("/freicar_"+agent_name+"_carla_proxy/spawn/heading", heading);

    std::uniform_real_distribution<float> generate_x(init_x, init_x);
    std::uniform_real_distribution<float> generate_y(init_y, init_y);
    heading = -heading * (M_PI / 180);
    std::uniform_real_distribution<float> generate_rot(heading, heading);

    for(int i=0;i<sizeOfParticles-1;i++){
        Particle particle;
        //Uniform weight to all particles in the beginning
        particle.weight = 1.0f;
        //Randomly select the x and y co-ordinate from the map
        float x = generate_x(generator_);
        float y = generate_y(generator_);
        //Select a random angle from -3.14 t0 +3.14
        float angle = generate_rot(generator_);

        Eigen::Transform<float, 3, Eigen::Affine> t = Eigen::Transform<float, 3, Eigen::Affine> ().Identity();
        t.translate(Eigen::Vector3f(x, y, 0.0));

        Eigen::Quaternionf rot;
        rot = Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitX())
                               * Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitY())
                               * Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ());
        t.rotate(rot);

        particle.transform = t;
        particles_.push_back(particle);
    }
    visualizer_->SendPoses(particles_, "particles", "map");
}

float particle_filter::getSpread(){
    float norm_val = 0;
    Eigen::Vector3f mean = Eigen::Vector3f(0, 0, 0);
    for(auto p_i = particles_.begin(); p_i != particles_.end(); p_i++){
        mean += p_i->weight * p_i->transform.translation();
        norm_val += p_i->weight;
    }
    mean /= norm_val;

    float spread = 0.;
    for(auto p_i = particles_.begin(); p_i != particles_.end(); p_i++){
        float position_error = p_i->weight * std::sqrt((mean - p_i->transform.translation()).squaredNorm());
        spread += position_error;
    }
    return spread/norm_val;
}

bool particle_filter::getQuality(float& quality){
    float avg_weight = 0.0f;

    if(!particle_memory_.empty() && memory_init_){
        avg_weight = 0.0f;

        for(auto i = particle_memory_.begin(); i != particle_memory_.end(); i++){
            avg_weight += *i;
        }
        avg_weight = avg_weight / particle_memory_.size();
        quality = avg_weight;
        return true;
    }

    quality = 1.0f;
    return false;

}

/*
 * Returns index of particle with highest weight
 */
int particle_filter::getBestParticleIndex(){
    //Particle best_p;
    float best_weight = std::numeric_limits<float>::min();
    int best_index = -1;
    for(size_t i = 0; i < particles_.size(); i++){
        if(particles_.at(i).weight > best_weight){
            best_weight = particles_.at(i).weight;
            best_index = static_cast<int>(i);
        }
    }
    return best_index;
}

/*
 * Returns index of particle with lowest weight
 */
int particle_filter::getWorstParticleIndex(){
    //Particle best_p;
    float worst_weight = std::numeric_limits<float>::max();
    int worst_index = -1;
    for(size_t i = 0; i < particles_.size(); i++){
        if(particles_.at(i).weight < worst_weight){
            worst_weight = particles_.at(i).weight;
            worst_index = static_cast<int>(i);
        }
    }
    return worst_index;
}

bool particleOrdering (Particle i,Particle j) { return (i.weight.load()<j.weight.load()); }

/*
 * Returns a particle with the rotation of the best particle and the mean position of the k_mean best particles.
 * TODO: Can be improved by computing mean rotation as well
 */
Particle particle_filter::getMeanParticle(int k_mean) {
    assert(k_mean <= particles_.size());
    std::vector<Particle> reverse_particles = particles_;
    std::sort(reverse_particles.begin(), reverse_particles.end(), particleOrdering);
    std::reverse(reverse_particles.begin(), reverse_particles.end());
    Eigen::Transform<float, 3, Eigen::Affine> out_t = Eigen::Transform<float, 3, Eigen::Affine>::Identity();
    Eigen::Matrix3f rot_total, rot_avg;
    rot_total << 0., 0., 0.,
            0., 0., 0.,
            0., 0., 0.;
    Eigen::Vector3f translation(0.0f, 0.0f, 0.0f);

    float avg_weight = 0.0;
    for (int i = 0; i < k_mean; i++) {
        if (i >= reverse_particles.size())
            break;

        translation += reverse_particles.at(i).transform.translation();
        rot_total += reverse_particles.at(i).transform.rotation();
        avg_weight += reverse_particles.at(i).weight;
    }

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            rot_avg(i, j) = rot_total(i, j) / k_mean;
        }
    }

    translation[0] = translation[0] / k_mean;
    translation[1] = translation[1] / k_mean;
    translation[2] = 0.;

    avg_weight = avg_weight / k_mean;

    out_t.translate(translation);
//  out_t.rotate(reverse_particles.at(0).transform.rotation());
    out_t.rotate(rot_avg);
    Particle out;
    out.weight = avg_weight;
    out.transform = out_t;

    return out;
}

/*
 * Returns particle with highest weight
 */
Particle particle_filter::getBestParticle(){
    {
        std::lock_guard<std::mutex> guard(particle_mutex_);
        size_t best_index = getBestParticleIndex();
        return particles_.at(best_index);
    }
}

/*
 * ##############################IMPLEMENT ME###########################################################################
 * Constant velocity motion model.
 * Takes given odometry (linear and angular velocities) and applies the motion (with added noise)
 * to every particle under the constant velocity assumption.
 * Params:
 * odometry : nav_msgs::odometry message that hold linear and angular velocity (odometry.twist.twist.linear.x,
 * odometry.twist.twist.linear.y, odometry.twist.twist.angular.z)
 * time_step: time in seconds that passed between the previous and the given odometry message
 *
 * The particles that should be moved are stored in "particles_"
 */
void particle_filter::ConstantVelMotionModel(nav_msgs::Odometry odometry, float time_step) {
    // Definitions of standard deviations for vel_x, vel_y and yaw_rate
    std::normal_distribution<float> dist_yaw(0.0, 0.2);
    std::normal_distribution<float> dist_x(0.0, 0.1);
    std::normal_distribution<float> dist_y(0.0, 0.1);

    for (size_t i = 0; i < particles_.size(); i++) {
        Eigen::Transform<float, 3, Eigen::Affine> particle_transform = particles_.at(i).transform;
        //Adding noise to get the new linear and angular velocity
        float x_vel = (odometry.twist.twist.linear.x - dist_x(generator_)) * time_step;
        float y_vel = (odometry.twist.twist.linear.y - dist_y(generator_)) * time_step;
        float z_vel = (odometry.twist.twist.angular.z - dist_yaw(generator_)) * time_step;
        // Creating an affine matrix and initializing it using the particles co-ordinates
        Eigen::Transform<float, 3, Eigen::Affine> t = Eigen::Transform<float, 3, Eigen::Affine>::Identity();
        t.translate(Eigen::Vector3f(x_vel, y_vel, 0.0f));
        Eigen::Quaternion<float> angular_vel;
        angular_vel = Eigen::AngleAxisf(0, Eigen::Vector3f::UnitX())
                * Eigen::AngleAxisf(0, Eigen::Vector3f::UnitY())
                * Eigen::AngleAxisf(z_vel, Eigen::Vector3f::UnitZ());
        t.rotate(angular_vel);

        particle_transform = particle_transform * t;
        particles_[i].transform = particle_transform;


    }
}

// This function initializes the KD-Trees for fast lookup of nearest neighbor data associations
bool particle_filter::InitMap(freicar::map::Map* map_data){
    if(map_data->status() != freicar::map::MapStatus::UNINITIALIZED){
        freicar::mapobjects::Pivot pivot = map_data->pivot();
        std::vector<freicar::mapobjects::Lane> lane_storage = map_data->getLanes();
        for(size_t i = 0;  i< lane_storage.size(); i++){
            freicar::mapobjects::Lane lane = lane_storage.at(i);
            if(lane.GetVisibility()){
                std::vector<Eigen::Vector3d> discretized_lane = map_helper::discretizeLane(lane, 0.05); // discretization step is in meter
                std::cout << "Discretized in " << discretized_lane.size() << std::endl;
                for(size_t i_p = 0; i_p < discretized_lane.size(); i_p++){
                    Eigen::Vector3d point = discretized_lane.at(i_p);
                    Point_KD<float> p_kd;
                    const float cm_p_m = 1.;
                    p_kd.x =  static_cast<float>(point.x() - pivot.GetPose().x()) / cm_p_m;
                    p_kd.y =  static_cast<float>(point.y() - pivot.GetPose().y()) / cm_p_m;

                    kd_data_.pts.push_back(p_kd);
                }
            }
        }


        // KD-Tree for signs
        std::vector<freicar::mapobjects::Roadsign> roadsigns = map_data->getSigns();

        for(size_t i = 0; i < roadsigns.size(); i++){
                freicar::mapobjects::Roadsign sign = roadsigns.at(i);
                std::cout << sign.GetSignType() << std::endl;

                Point_KD<float> p_kd;
                p_kd.x =  static_cast<float>(sign.GetPosition().x() - pivot.GetPose().x()) / 1.0f;
                p_kd.y =  static_cast<float>(sign.GetPosition().y() - pivot.GetPose().y()) / 1.0f;

                if(sign_data_.find(sign.GetSignType()) == sign_data_.end()){
                    // Create new KD data
                    sign_data_[sign.GetSignType()] = PointCloud<float>();
                }
                sign_data_.at(sign.GetSignType()).pts.push_back(p_kd);

                std::cout << "Added sign " << sign.GetSignType() << " at x: " << p_kd.x << " y: " << p_kd.y << std::endl;
        }

        return true;
    }
    return false;
}

// This function transforms all sign positions from /zed_camera to /base_link
std::vector<Sign> particle_filter::transformSignsToCarBaseLink(const std::vector<Sign>& signs){

    static bool initialized = false;
    static tf::StampedTransform c_t_b;

    static tf::TransformListener listener;

    if(signs.size()>0){
        for(int i=0; i<signs.size();i++){
            std::cout<<"observed sign in zed camera frame:"<<signs.at(i).position.x()<<" "<<signs.at(i).position.y()<<std::endl;
        }
    }

    if(!initialized){
        try{
            listener.lookupTransform(agent_name+"/base_link", agent_name+"/zed_camera",
                                     ros::Time(0), c_t_b);
            initialized = true;
        }
        catch (tf::TransformException ex){
            ROS_ERROR("%s",ex.what());
            ros::Duration(1.0).sleep();
        }
    }

    std::vector<Sign> out;

    if(initialized){
        Eigen::Affine3d t_camera_car_d;
        tf::transformTFToEigen(c_t_b, t_camera_car_d);
        Eigen::Affine3f t_camera_car = t_camera_car_d.cast <float> ();

        for(size_t i =0; i < signs.size(); i++){
            const Sign& s = signs.at(i);
            Sign new_s = s;
            new_s.position = t_camera_car * s.position;
            tf::Vector3 a(s.position.x(), s.position.y(), s.position.z());
            a = c_t_b * a;
            std::cout<<"transform from camera to car:"<<a.x()<<" "<<a.y()<<std::endl;
            out.push_back(new_s);
        }
    }

    return out;
}

// Normalizes all particle weights by the given value
void particle_filter::NormalizeParticles(const float norm_val){
    // Normalize weights
    for(auto p_i = particles_.begin(); p_i != particles_.end(); p_i++){
        p_i->weight = p_i->weight.load() / norm_val;
    }
}
/*
 Main function for starting the observation step (sensor model).
 Calculates all pose probabilities for all particles (using sensor model class), normalizes particle weights,
 resamples particles and detects wether or not the filter delocalized
 */
bool particle_filter::ObservationStep(const std::vector<cv::Mat> reg, const std::vector<Sign> observed_signs){
    const std::vector<Sign> car_signs = transformSignsToCarBaseLink(observed_signs); // inline

    // We only want to resample if everything is initialized and if we drive at least 0.1 m/s.
    if(odo_init_ && particles_init_ && abs(latest_x_vel) >= 0.05){
//    if(odo_init_ && particles_init_ ){
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        float best_val =  0.0;
        bool sensor_model_success = sensor_model_->calculatePoseProbability(reg, car_signs, particles_, best_val);
        if(!sensor_model_success)
            return false;

        // Fifo best value memory
        particle_memory_.push_front(best_val);
        particle_memory_.pop_back();
        memory_insert_cnt_++;
        if(memory_insert_cnt_ >= BEST_PARTICLE_HISTORY){
            memory_init_ = true;
        }

        this->NormalizeParticles(best_val);

        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>( t2 - t1 ).count();
        std::cout << "Sensor model took: " << duration << " milliseconds" << std::endl;
        // lock scope
        {
            std::lock_guard<std::mutex> guard(particle_mutex_);

            float quality = 1.0f;
            if(getQuality(quality) || best_val == 0){
                std::cout << "QUALITY: " << quality << std::endl;
                if(quality < QUALITY_RELOC_THRESH || best_val == 0){
                    std::cerr << "DELOCALIZED! REINITIALIZING!" << std::endl;
                    InitParticles();
                    return true;
                }
            }
            // Resample new particles
            LowVarianceSampling();
            //RouletteSampling();
        }
    }

    return true;
}

/*
 * This function applies the motion model and moves all particles given the odometry
 */
void particle_filter::MotionStep(nav_msgs::Odometry odometry){
    // We need two incoming transform to start the particle filter

    if(!odo_init_){
        prev_odo_ = odometry;
        odo_init_ = true;
        return;
    }

    // Apply motion model to particles...
    latest_x_vel = odometry.twist.twist.linear.x;
    {
        std::lock_guard<std::mutex> guard(particle_mutex_);
        ConstantVelMotionModel(odometry, abs((odometry.header.stamp - prev_odo_.header.stamp).toSec()));
        visualizer_->SendPoses(particles_, "particles", "map");
    }

    prev_odo_ = odometry;
}

PointCloud<float>& particle_filter::getMapKDPoints(){
    return kd_data_;
}
