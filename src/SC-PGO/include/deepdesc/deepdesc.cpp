#include "deepdesc/deepdesc.h"
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/statistical_outlier_removal.h>

pcl::PointCloud<PointType> DeepDescManager::data_preprocess(pcl::PointCloud<PointType>::Ptr input) {
    // 1. Removing outliers using a StatisticalOutlierRemoval filter
    pcl::PointCloud<PointType>::Ptr cloud_filtered (new pcl::PointCloud<PointType>);
    // cloud_filtered = input;

    pcl::StatisticalOutlierRemoval<PointType> sor;
    sor.setInputCloud(input);
    sor.setMeanK(16);
    sor.setStddevMulThresh(2.0); 
    sor.filter(*cloud_filtered);

    // // RadiusOutlierRemoval
    // pcl::RadiusOutlierRemoval<PointType> outrem;
    // outrem.setInputCloud(input);
    // outrem.setRadiusSearch(3.0);
    // outrem.setMinNeighborsInRadius(5);
    // outrem.filter(*cloud_filtered);
  
    // 2. Use RandomSample to downsample the pointcloud to get the 4096 size pcl.
    pcl::PointCloud<PointType>::Ptr res (new pcl::PointCloud<PointType>);
    pcl::RandomSample<PointType> rs;
    rs.setInputCloud(cloud_filtered);
    rs.setSample(numPoints_);
    rs.filter(*res);
    return *res;
}

// pcl::PointCloud<PointType> DeepDescManager::data_preprocess(pcl::PointCloud<PointType>::Ptr input) {
//     // 1. Removing outliers using a StatisticalOutlierRemoval filter
//     pcl::PointCloud<PointType>::Ptr cloud_filtered (new pcl::PointCloud<PointType>);
//     pcl::StatisticalOutlierRemoval<PointType> sor;
//     sor.setInputCloud(input);
//     sor.setMeanK(16);
//     sor.setStddevMulThresh(2.0); 
//     sor.filter(*cloud_filtered);

//     // 2. Manual random downsampling (simpler version)
//     pcl::PointCloud<PointType>::Ptr res(new pcl::PointCloud<PointType>);
    
//     // int numPoints_ = 4096;  // Set the number of points you want in the downsampled point cloud
//     std::vector<int> sampled_indices;  // To store selected indices
//     std::srand(static_cast<unsigned int>(std::time(0)));  // Seed random number generator

//     // Ensure we don't sample more points than available
//     numPoints_ = std::min(numPoints_, static_cast<int>(cloud_filtered->points.size()));

//     // Randomly select indices
//     while (sampled_indices.size() < numPoints_) {
//         int index = std::rand() % cloud_filtered->points.size();  // Random index
//         // Only add unique indices (no duplicates)
//         if (std::find(sampled_indices.begin(), sampled_indices.end(), index) == sampled_indices.end()) {
//             sampled_indices.push_back(index);  // Add index if not already selected
//         }
//     }

//     // Directly add the selected points to the result cloud (no sorting needed)
//     for (int index : sampled_indices) {
//         res->push_back(cloud_filtered->points[index]);
//     }

//     return *res;
// }

void DeepDescManager::makeAndSaveDeepDescAndKeys( pcl::PointCloud<PointType> _scan_down )
{   
    TicTocV2 t_data_preprocess(false);
    // convert from pcl::PointXYZI to std::vector<float>
    std::vector<float> scan_data;
    // std::cout << "scan size1: " << _scan_down.points.size() << std::endl;
    // std::chrono::system_clock::time_point start_time = std::chrono::system_clock::now();
    _scan_down = data_preprocess(_scan_down.makeShared());
    // std::cout << "scan size2: " << _scan_down.points.size() << std::endl;
    // std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
    // std::chrono::duration<double> data_preprocess_time = end_time - start_time; 
    // std::cout << "data_preprocess time: " << data_preprocess_time.count() << std::endl;

    scan_data.resize(numPoints_ * inputDim_, 0.0);
    #ifdef MP_EN
        omp_set_num_threads(3);
        #pragma omp parallel for
    #endif
    for (size_t i = 0; i < numPoints_; i++) {
        scan_data[inputDim_ * i + 0] = _scan_down.points[i].x;
        scan_data[inputDim_ * i + 1] = _scan_down.points[i].y;
        scan_data[inputDim_ * i + 2] = _scan_down.points[i].z;
    }
    t_data_preprocess.toc("Data preprocess");


    TicTocV2 t_make_desc(false);
    // make DeepDesc and save
    std::vector<float> query_des_vector;
    pointnetvlad_.compute(scan_data, query_des_vector);
    // for (size_t i = 0; i < outputDim_; i++) {
    //     std::cout << query_des_vector[i] << " ";
    // }
    // std::cout << std::endl;
    deepdesc_invkeys_mat_.push_back(query_des_vector);
    t_make_desc.toc("DeepDesc make");
} // DeepDescManager::makeAndSaveDeepDescAndKeys

void DeepDescManager::remakeAndSaveDeepDescAndKeysAtIndex(pcl::PointCloud<PointType> _scan_down, int idx)
{   
    TicTocV2 t_data_preprocess(false);
    // convert from pcl::PointXYZI to std::vector<float>
    std::vector<float> scan_data;
    // std::cout << "scan size1: " << _scan_down.points.size() << std::endl;
    _scan_down = data_preprocess(_scan_down.makeShared());
    // std::cout << "scan size2: " << _scan_down.points.size() << std::endl;
    scan_data.resize(numPoints_ * inputDim_, 0.0);
    #ifdef MP_EN
        omp_set_num_threads(3);
        #pragma omp parallel for
    #endif
    for (size_t i = 0; i < numPoints_; i++) {
        scan_data[inputDim_ * i + 0] = _scan_down.points[i].x;
        scan_data[inputDim_ * i + 1] = _scan_down.points[i].y;
        scan_data[inputDim_ * i + 2] = _scan_down.points[i].z;
    }
    t_data_preprocess.toc("Data preprocess");


    TicTocV2 t_make_desc(false);
    // make DeepDesc and save
    std::vector<float> query_des_vector;
    pointnetvlad_.compute(scan_data, query_des_vector);

    if(idx<deepdesc_invkeys_mat_.size() && idx>=0){
        deepdesc_invkeys_mat_[idx] = query_des_vector;
    }
    else if(idx==deepdesc_invkeys_mat_.size() || idx==-1){
        deepdesc_invkeys_mat_.push_back(query_des_vector);
    }
    else{
        std::cout<<"idx out of range"<<std::endl;
    }
    t_make_desc.toc("DeepDesc make");
} // DeepDescManager::remakeAndSaveDeepDescAndKeysAtIndex

std::pair<int, float> DeepDescManager::detectLoopClosureID ( void )
{
    int tree_build_offset = 0;
    // tree_build_offset = TREE_MAKE_STEP_/2;

    int loop_id { -1 }; // init with -1, -1 means no loop.

    auto curr_key = deepdesc_invkeys_mat_.back(); // current observation (query)

    /** step 1: candidates from ringkey tree_ */
    if( (int)deepdesc_invkeys_mat_.size() < NUM_EXCLUDE_RECENT + 1 + tree_build_offset)
    {
        std::pair<int, float> result {loop_id, 0.0};
        return result; // Early return 
    }

    // tree_ reconstruction (not mandatory to make everytime)
    // std::cout << "tree_making_period_conter: " << tree_making_period_conter << std::endl;
    if( tree_making_period_conter % TREE_MAKING_PERIOD_ == 0) // to save computation cost
    {
        TicTocV2 t_tree_construction;

        deepdesc_invkeys_to_search_.clear();
        // deepdesc_invkeys_to_search_.assign( deepdesc_invkeys_mat_.begin(), deepdesc_invkeys_mat_.end() - NUM_EXCLUDE_RECENT) ;
        #ifdef MP_EN
            omp_set_num_threads(3);
            #pragma omp parallel for
        #endif
        for (auto iter = deepdesc_invkeys_mat_.begin()+tree_build_offset; 
                  iter < deepdesc_invkeys_mat_.end() - NUM_EXCLUDE_RECENT; 
                  iter += TREE_MAKE_STEP_) {
            deepdesc_invkeys_to_search_.push_back(*iter);
        }

        if (deepdesc_invkeys_to_search_.empty())
        {
            std::pair<int, float> result {loop_id, 0.0};
            return result; // Early return 
        }

        deepdesc_tree_.reset(); 
        deepdesc_tree_ = std::make_unique<InvKeyTree>(outputDim_ /* dim */, deepdesc_invkeys_to_search_, 10 /* max leaf */ );
        
        t_tree_construction.toc("Tree construction");
    }
    tree_making_period_conter = tree_making_period_conter + 1;
        
    double min_dist = 10000000; // init with somthing large
    double second_min_dist = 10000000; // init with somthing large
    int nn_align = 0;
    int nn_idx = 0;
    float yaw_diff_rad = 0;

    // knn search
    std::vector<size_t> candidate_indexes( NUM_CANDIDATES_FROM_TREE, -1.0); 
    std::vector<float> out_dists_sqr( NUM_CANDIDATES_FROM_TREE );

    TicTocV2 t_tree_search;
    nanoflann::KNNResultSet<float> knnsearch_result( NUM_CANDIDATES_FROM_TREE );
    knnsearch_result.init( &candidate_indexes[0], &out_dists_sqr[0] );
    deepdesc_tree_->index->findNeighbors( knnsearch_result, &curr_key[0] /* query */, nanoflann::SearchParams(10) );
    // std::cout<< "knnsearch_result.size(): " << knnsearch_result.size() << std::endl;
    t_tree_search.toc("Tree search");

    // std::cout<<"*************************************\n";
    // std::cout<< "curr_key-" << deepdesc_invkeys_mat_.size()-1 << ": ";
    for( int candidate_iter_idx = 0; candidate_iter_idx < knnsearch_result.size(); candidate_iter_idx++ )
    {
        int candidate_idx = TREE_MAKE_STEP_*candidate_indexes[candidate_iter_idx] + tree_build_offset;
        float candidate_dist = out_dists_sqr[candidate_iter_idx];

        if(candidate_dist == 0.0 || candidate_idx == -1){
            continue;;
        }

        int index_diff = abs(int(deepdesc_invkeys_mat_.size() - 1 - candidate_idx));
        if(index_diff > 1*NUM_EXCLUDE_RECENT)
        {
            // std::cout<< candidate_idx << "(" << candidate_dist << ") ";
            if(candidate_dist < min_dist){
                // std::cout<<"cadi-cur: "<<candidate_idx<<" "<<deepdesc_invkeys_mat_.size()<<std::endl;
                second_min_dist = min_dist;  // not used, because out_dists_sqr is sorted (little to large).
                min_dist = candidate_dist;
                nn_idx = candidate_idx;
            }
        }
    }
    // std::cout<<std::endl;

    // for (size_t i = 0; i < out_dists_sqr.size(); i++)
    // {
    //     if (out_dists_sqr[i] == 0.0)
    //     {
    //         std::cout << "[single agnet_ID: " << candidate_indexes[i] << ", Dist: " << out_dists_sqr[i] << "] ";
    //     }
    // }
    // std::cout << std::endl;

    // std::cout << "min_dist: " << min_dist <<" "<<second_min_dist<< std::endl;
    // std::cout<< "min/seceond_min: " << out_dists_sqr[0]/out_dists_sqr[1] << std::endl;

    if(min_dist > DeepLOOP_DIST_THRES){
        return std::pair<int, float>(loop_id, yaw_diff_rad);
        // std::cout<<" because: min_dist=" << min_dist << " too large."<<std::endl;
    }
    else if(knnsearch_result.size()>=2 && (out_dists_sqr[0]/out_dists_sqr[1] > LOOP_MIN_SECOND_RATE_THRES)){
        // std::cout<<"cadidate invalid because: min/seceond_min=" <<out_dists_sqr[0]/out_dists_sqr[1]<<" too large."<<std::endl;
    }
    else if(knnsearch_result.size()==1){
        loop_id = nn_idx;
        // std::cout<<"warning: cadidate used but only one candidate."<<std::endl;
    }
    else{
        loop_id = nn_idx;
        // std::cout<<"cadidate "<< nn_idx <<" valid: "<<min_dist<<", "<<out_dists_sqr[0]/out_dists_sqr[1]<<"."<<std::endl;
    }
    // std::cout<<"*************************************\n";

    // std::cout<<"cur-cadi: "<<deepdesc_invkeys_mat_.size()-1<<" "<<loop_id<<std::endl;

    // nn_idx = candidate_indexes[0];
    // std::pair<int, float> result {nn_idx, yaw_diff_rad};
    // std::cout << "nn_idx: " << nn_idx << std::endl;

    std::pair<int, float> result {loop_id, yaw_diff_rad};
    return result;

    // /* 
    //  *  step 2: pairwise distance (find optimal columnwise best-fit using cosine distance)
    //  */
    // TicTocV2 t_calc_dist;   
    // for ( int candidate_iter_idx = 0; candidate_iter_idx < NUM_CANDIDATES_FROM_TREE; candidate_iter_idx++ )
    // {
    //     MatrixXd polarcontext_candidate = polarcontexts_[ candidate_indexes[candidate_iter_idx] ];
    //     std::pair<double, int> sc_dist_result = distanceBtnScanContext( curr_desc, polarcontext_candidate ); 
        
    //     double candidate_dist = sc_dist_result.first;
    //     int candidate_align = sc_dist_result.second;

    //     if( candidate_dist < min_dist && abs(double(candidate_indexes[candidate_iter_idx]-polarcontexts_.size()-1))>200)
    //     {
    //         min_dist = candidate_dist;
    //         nn_align = candidate_align;

    //         nn_idx = candidate_indexes[candidate_iter_idx];
    //     }
    // }
    // t_calc_dist.toc("Distance calc");

    // /* 
    //  * loop threshold check
    //  */
    // if( min_dist < SC_DIST_THRES )
    // {
    //     loop_id = nn_idx; 
    //     // std::cout.precision(3); 
    //     // cout << "[Loop found] Nearest distance: " << min_dist << " btn " << polarcontexts_.size()-1 << " and " << nn_idx << "." << endl;
    //     // cout << "[Loop found] yaw diff: " << nn_align * PC_UNIT_SECTORANGLE << " deg." << endl;
    // }
    // else
    // {
    //     std::cout.precision(3); 
    //     // cout<< "SC_DIST_THRES: "<< SC_DIST_THRES << endl;
    //     // cout << "[Not loop] Nearest distance: " << min_dist << " btn " << polarcontexts_.size()-1 << " and " << nn_idx << "." << endl;
    //     // cout << "[Not loop] yaw diff: " << nn_align * PC_UNIT_SECTORANGLE << " deg." << endl;
    // }

    // // To do: return also nn_align (i.e., yaw diff)
    // float yaw_diff_rad = deg2rad(nn_align * PC_UNIT_SECTORANGLE);
    // std::pair<int, float> result {loop_id, yaw_diff_rad};

    // return result;

} // DeepDescManager::detectLoopClosureID