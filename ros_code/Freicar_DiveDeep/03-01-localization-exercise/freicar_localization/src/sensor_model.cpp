/*
 * Author: Johan Vertens (vertensj@informatik.uni-freiburg.de)
 * Project: FreiCAR
 * Do NOT distribute this code to anyone outside the FreiCAR project
 */

#include "sensor_model.h"
#include "particle_filter.h"

float getEuclideanDistance(float a[2], float b[2]);

std::unordered_set<int> pickSet(int N, int k, std::mt19937& gen)
{
    std::unordered_set<int> elems;
    for (int r = N - k; r < N; ++r) {
        int v = std::uniform_int_distribution<>(1, r)(gen);

        if (!elems.insert(v).second) {
            elems.insert(r);
        }
    }
    return elems;
}

/*
 * Returns k random indeces between 0 and N
 */
std::vector<int> pick(int N, int k) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::unordered_set<int> elems = pickSet(N, k, gen);

    std::vector<int> result(elems.begin(), elems.end());
    std::shuffle(result.begin(), result.end(), gen);
    return result;
}

/*
 * Constructor of sensor model. Builds KD-tree indeces
 */
sensor_model::sensor_model(PointCloud<float> map_data, std::map<std::string, PointCloud<float> > sign_data, std::shared_ptr<ros_vis> visualizer, bool use_lane_reg):map_data_(map_data), sign_data_(sign_data), map_index_(2 /*dim*/, map_data_, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) ), visualizer_(visualizer)
{
    // Creating KD Tree indeces for fast nearest neighbor search
    map_index_.buildIndex();
    for(auto ds = sign_data_.begin(); ds != sign_data_.end(); ds++){
        std::cout << "Creating kd tree for sign type: " << ds->first << " with " << ds->second.pts.size() << " elements..." << std::endl;
        sign_indeces_[ds->first] = std::unique_ptr<map_kd_tree>(new map_kd_tree(2, ds->second, nanoflann::KDTreeSingleIndexAdaptorParams(10)));
        sign_indeces_[ds->first]->buildIndex();
    }
    use_lane_reg_ = use_lane_reg;
}

/*
 * For any given observed lane center points return the nearest lane-center points in the map
 * This is using a KD-Tree for efficient match retrieval
 */
std::vector<Eigen::Vector3f> sensor_model::getNearestPoints(std::vector<Eigen::Vector3f> sampled_points){
    // Get data association
    assert(map_data_.pts.size() > 0);
    std::vector<Eigen::Vector3f> corr_map_associations;
    for(size_t i=0; i < sampled_points.size(); i++){
        // search nearest neighbor for sampled point in map
        float query_pt[2] = { static_cast<float>(sampled_points.at(i).x()), static_cast<float>(sampled_points.at(i).y())};

        const size_t num_results = 1;
        size_t ret_index;
        float out_dist_sqr;
        nanoflann::KNNResultSet<float> resultSet(num_results);
        resultSet.init(&ret_index, &out_dist_sqr );
        map_index_.findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

        // Gather map vector
        Point_KD<float> corr_p = map_data_.pts.at(ret_index);
        corr_map_associations.push_back(Eigen::Vector3f((corr_p.x), (corr_p.y), 0.0));
    }
    return corr_map_associations;
}

/*
 * For given observed signs return nearest sign positions with the same type.
 * This is using a KD-Tree for efficient match retrieval.
 * Returns a empty vector if not possible
 */
std::vector<Eigen::Vector3f> sensor_model::getNearestPoints(std::vector<Sign> observed_signs){
    // Get data association
    std::vector<Eigen::Vector3f> corr_map_associations;
    for(size_t i=0; i < observed_signs.size(); i++){
        const Sign& s = observed_signs.at(i);
        if(sign_indeces_.find(s.type) != sign_indeces_.end()){
            // search nearest neighbor for sampled point in map
            float query_pt[2] = {s.position[0], s.position[1]};

            const size_t num_results = 1;
            size_t ret_index;
            float out_dist_sqr;
            nanoflann::KNNResultSet<float> resultSet(num_results);
            resultSet.init(&ret_index, &out_dist_sqr );
            sign_indeces_[s.type]->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

            // Gather sign position from map
            if(out_dist_sqr < 1e30){
                Point_KD<float> corr_p = sign_data_[s.type].pts.at(ret_index);
                corr_map_associations.push_back(Eigen::Vector3f((corr_p.x), (corr_p.y), 0.0));
            }else{
                std::cerr << "Invalid query..." << std::endl;
                return std::vector<Eigen::Vector3f>();
            }
        }else{
            std::cerr << "No corrensponding sign in map kd indeces..." << std::endl;
            return std::vector<Eigen::Vector3f>();
        }
    }
    return corr_map_associations;
}

/*
 * Transforms a given list of 3D points by a given affine transformation matrix
 */
std::vector<Eigen::Vector3f> sensor_model::transformPoints(const std::vector<Eigen::Vector3f> points, const Eigen::Transform<float,3,Eigen::Affine> transform){
    std::vector<Eigen::Vector3f> transformed;
    for(size_t i =0; i < points.size(); i++){
        Eigen::Vector3f p_world = transform * points.at(i);
        transformed.push_back(p_world);
    }
    return transformed;
}

/*
 * Returns sum of given float-vector
 */
float sensor_model::sumWeights(const std::vector<float>& weights){
    float sum = 0.0f;
    for(auto i = weights.begin(); i != weights.end(); i++){
        sum += *i;
    }
    return sum;
}

/*
 * Transforms sign position by a given affine transformation matrix
 */
std::vector<Sign> sensor_model::transformSigns(const std::vector<Sign>& signs, const Eigen::Transform<float,3,Eigen::Affine>& particle_pose){
    std::vector<Sign> transformed_signs;
    for(size_t i =0; i < signs.size(); i++){
        const Sign& s = signs.at(i);
        Sign t_s = s;
        t_s.position = particle_pose * s.position;
        transformed_signs.push_back(t_s);
    }
    return transformed_signs;
}

/*
 * ##########################IMPLEMENT ME###############################################################################
 * Sensor-model. This function does the following:
 * --Calculates the likelihood of every particle being at its respective pose.
 * The likelihood should be stored in the particles weight member variable
 * The observed_signs variable contain all observed signs at the current timestep. They are relative to freicar_X/base_link.
 * The current particles are given with the variable "particles"
 * The true positions of all signs for a given type are stored in: sign_data_[observed_signs.at(i).type].pts , where
 * observed_signs.at(i).type is the sign_type of the i'th observed sign and pts is a array of positions (that have
 * the member x and y)
 * For lane regression data: The function getNearestPoints() might come in handy for getting the closest points to the
 * sampled and observed lane center points.
 *
 * The variable max_prob must be filled with the highest likelihood among all particles. If the average
 * of the last BEST_PARTICLE_HISTORY (defined in particle_filter.h) max_prob values is under the value
 * QUALITY_RELOC_THRESH (defined in particle_filter.h) a resampling will be initiated. So you may want to adjust the threshold.
 *
 * The function needs to return True if everything was successfull and False otherwise.

 */
bool sensor_model::calculatePoseProbability(const std::vector<cv::Mat> lane_regression, const std::vector<Sign> observed_signs, std::vector<Particle>& particles, float& max_prob){
    // Check if there is a bev lane regression matrix available. If so, use it in the observation step
    std::cout<<"______________________________________________________"<<std::endl;
    max_prob = 0.0; // Dummy for compilation
    bool success = false; // Dummy for compilation

    // Hint: The following code line can be used for transforming the sign positions using the particles pose.
    float total_weight = 0;
    if(observed_signs.size()>0){
        for(int i=0; i<observed_signs.size();i++){
            std::cout<<"observed sign in car frame:"<<observed_signs.at(i).position.x()<<" "<<observed_signs.at(i).position.y()<<std::endl;
        }
    }

        for(int j=0;j<particles.size();j++){
            for(int i=0;i<observed_signs.size();i++){
                //Transform the observed signs from freicar_x/base_link to map frame using the transform from the any particle present in map frame.
                const std::vector<Sign> observed_world_signs = transformSigns(observed_signs, particles[j].transform);
                //Get the real co-ordinates of the signs from the map which are nearest to observed signs.
                std::vector<Eigen::Vector3f> gt_nearest_signs = getNearestPoints(observed_world_signs);
                float observed_sign[] = {observed_world_signs.at(i).position.x(), observed_world_signs.at(i).position.y()};
                float probability = 0;
                float particle_pos[] = {particles[j].transform(0,3), particles[j].transform(1,3)};
                std::vector<float> shortest_dist_sign = getSignNearestToParticle(particle_pos, gt_nearest_signs);
                float particle_short_dist_sign[] = {shortest_dist_sign[0], shortest_dist_sign[1]};
                std::cout<<"----------------------------------------------------------------------"<<std::endl;
                std::cout<<"particle pos:"<<particle_pos[0]<<" "<<particle_pos[1]<<std::endl;
                std::cout<<"observed sign in map:"<<observed_world_signs.at(i).position.x()<<" "<<observed_world_signs.at(i).position.y()<<std::endl;
                if(!gt_nearest_signs.empty()){
                    for(int k=0;k<gt_nearest_signs.size();k++){
                        std::cout<<"gt_sign:"<<gt_nearest_signs.at(k).x()<<" "<<gt_nearest_signs.at(k).y()<<" "<<gt_nearest_signs.at(k).z()<<std::endl;
                        float dist_k_to_shortest_sign = getEuclideanDistance(gt_nearest_signs[k].data(), particle_short_dist_sign);
                        float probability_of_each_sign = probability_density_function(0.0, 0.3, dist_k_to_shortest_sign);
                        float dist_observedSign_to_gt = getEuclideanDistance(observed_sign, gt_nearest_signs[k].data());
                        float probability_of_observed_sign = probability_density_function(0.0, 0.3, dist_observedSign_to_gt);
                        std::cout<<"probability_of_observed_sign:"<<probability_of_observed_sign<<" "<<"probability_of_each_sign:"<<probability_of_each_sign<<std::endl;
                        probability += probability_of_observed_sign * probability_of_each_sign;
                    }
                    success = true;
                }
            particles[j].weight = probability;
                std::cout<<"probability:"<<probability<<std::endl;
            total_weight += particles[j].weight;
        }
    }
    //normalising the weights
    for(int i=0;i<particles.size();i++){
        particles[i].weight = particles[i].weight/total_weight;
        if(particles[i].weight>max_prob){
            max_prob = particles[i].weight;
        }
    }

    Particle best_particle = getBestParticle(particles);
    std::cout<<"get best particle:"<<best_particle.weight<<" "<<best_particle.transform(0,3)<<" "<<best_particle.transform(1,3)<<std::endl;

    return success;
}

float sensor_model::probability_density_function(float m, float s, float x){
    return ( 1 / ( s * sqrt(2*M_PI) ) ) * exp( -0.5 * pow( (x-m)/s, 2.0 ) );
}

std::vector<float> sensor_model::getSignNearestToParticle(float *x, std::vector<Eigen::Vector3f> gt_sign) {
    float shortest_distance = RAND_MAX;
    float pos[2], map[2];
    std::vector<float> sign;
    pos[0] = x[0]; pos[1] = x[1];
    for(int i=0;i<gt_sign.size();i++){
        map[0] = gt_sign[i].x(); map[1] = gt_sign[i].y();
        float dst = getEuclideanDistance(pos, map);
        if(dst<=shortest_distance){
            shortest_distance = dst;
            sign.push_back(gt_sign[i].x());
            sign.push_back(gt_sign[i].y());
            sign.push_back(shortest_distance);
            sign.push_back(i);
        }
    }
    return sign;
}

Particle sensor_model::getBestParticle(std::vector<Particle> &particles) {
    //Particle best_p;
    float best_weight = std::numeric_limits<float>::min();
    int best_index = -1;
    for(size_t i = 0; i < particles.size(); i++){
        if(particles.at(i).weight > best_weight){
            best_weight = particles.at(i).weight;
            best_index = static_cast<int>(i);
        }
    }
    return particles[best_index];
}

float getEuclideanDistance(float a[2], float b[2]) {
    return sqrt(pow(a[0]-b[0],2) + pow(a[1]-b[1], 2));
}