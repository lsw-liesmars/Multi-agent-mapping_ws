#ifndef DEEPDESC_H
#define DEEPDESC_H

#include <iostream>
#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <torch/script.h>
#include <torch/torch.h>

#include "scancontext/nanoflann.hpp"
#include "scancontext/KDTreeVectorOfVectorsAdaptor.h"
#include "aloam_velodyne/tic_toc.h"

typedef pcl::PointXYZI PointType;

class PointNetVlad {
private:
    torch::jit::script::Module module;
public:
    int num_points_ = 4096;
    int input_dim_ = 3;
    int output_dim_ = 256;

public:
    PointNetVlad(const std::string &model_path) {
        module = torch::jit::load(model_path);
        module.eval();
        std::cout << "PointNetVlad model loaded from " << model_path << std::endl;
    }

    PointNetVlad(const std::string &model_path, int num_points=4096, int input_dim=3, int output_dim=256) : 
    num_points_(num_points), input_dim_(input_dim), output_dim_(output_dim)
    {
        module = torch::jit::load(model_path);
        module.eval();
        std::cout << "PointNetVlad model loaded from " << model_path << std::endl;
    }

    PointNetVlad() {}
    ~PointNetVlad() {}

    void compute(std::vector<float> &input, std::vector<float> &descriptor) {
        // TicTocV2 t_build(false);
        // torch::Tensor tester = torch::from_blob(input.data(), {1, 1, (int)num_points_, (int)input_dim_}, torch::kFloat);
        // torch::Tensor query_des = module.forward({tester}).toTensor();
        // t_build.toc("Build query desc");

        // TicTocV2 t_convert(false);
        // // Convert tensor to std::vector<float>
        // query_des = query_des.view({-1});
        // descriptor = std::vector<float>(query_des.data_ptr<float>(), query_des.data_ptr<float>() + query_des.numel());
        // t_convert.toc("Convert tensor to std::vector<float>");
        
        torch::DeviceType device_type;
        device_type = torch::kCPU;
        torch::Device device(device_type);
        // std::cout<<"cuda support:"<< (torch::cuda::is_available()?"ture":"false")<<std::endl;
        torch::Tensor queryDescriptor;
        // generate the global descriptor of the query
        torch::Tensor testeri = torch::from_blob(input.data(), {1, 1, (int)num_points_, (int)input_dim_}, torch::kFloat).to(device);
        torch::Tensor result = module.forward({testeri}).toTensor();
        queryDescriptor = result.to(torch::kCPU);
        descriptor.resize(output_dim_);
        #ifdef MP_EN
            omp_set_num_threads(3);
            #pragma omp parallel for
        #endif
        for (int j = 0; j < output_dim_; j++) {
            descriptor[j] = queryDescriptor[0][j].item<float>();
            // std::cout << descriptor[j] << " ";
        }
        // std::cout << std::endl;
    }

    template <typename PointT>
    void compute(typename pcl::PointCloud<PointT>::ConstPtr queryscan, std::vector<float> &descriptor) {
        if (queryscan->size() != num_points_) {
            std::cerr << "query scan size " << queryscan->size() << " not equal to num_points "
                      << num_points_ << ". Downsample before calling compute()." << std::endl;
            return;
        }
        // convert pointcloud to tensor
        torch::Tensor tester = torch::zeros({1, 1, (int)num_points_, (int)input_dim_}, torch::kFloat);
        for(size_t i=0; i<queryscan->size(); i++) {
            tester[0][0][i][0] = queryscan->points[i].x;
            tester[0][0][i][1] = queryscan->points[i].y;
            tester[0][0][i][2] = queryscan->points[i].z;
        }
        // gen query descriptor
        torch::Tensor query_des = module.forward({tester}).toTensor();

        // Convert tensor to std::vector<float>
        query_des = query_des.view({-1});
        descriptor = std::vector<float>(query_des.data_ptr<float>(), query_des.data_ptr<float>() + query_des.numel());
    }
};

using KeyMat = std::vector<std::vector<float> >;
using InvKeyTree = KDTreeVectorOfVectorsAdaptor< KeyMat, float >;
class DescriptorDB {
private:
    std::unique_ptr<InvKeyTree> database_tree_;
    KeyMat des_to_search_;

public:
    static const size_t output_dim;
    explicit DescriptorDB(const KeyMat &database_des) {
        des_to_search_ = database_des;
        database_tree_ = std::make_unique<InvKeyTree>(output_dim, des_to_search_, 10);
    }

    DescriptorDB() {}

    template <typename PointT>
    void initializeFromScans(const std::vector<typename pcl::PointCloud<PointT>::ConstPtr> &scans, const std::string &model_path) {
        // create train Database and generate descriptors
        std::cout << "create train database" <<std::endl;
        PointNetVlad pointnetvlad(model_path);
        des_to_search_.clear();
        des_to_search_.reserve(scans.size());
        for (int i = 0; i < scans.size(); ++i) {
            std::vector<float> descriptor;
            pointnetvlad.compute(scans[i], descriptor);
            des_to_search_.emplace_back(descriptor);
        }
        database_tree_ = std::make_unique<InvKeyTree>(output_dim, des_to_search_, 10);
    }

    /**
     * query database to find candidate loops
     * return number of valid loops
     */
    size_t query(const std::vector<float> &query_des, std::vector<size_t> &candidate_indexes, 
            std::vector<float> &out_dists_sq, int maxcands, float dist_threshold) {
        int capacity = maxcands;
        candidate_indexes.resize(capacity);
        out_dists_sq.resize(capacity);
        nanoflann::KNNResultSet<float> knnsearch_result(capacity);
        knnsearch_result.init(&candidate_indexes[0], &out_dists_sq[0]);
        database_tree_->index->findNeighbors(knnsearch_result, &query_des[0], nanoflann::SearchParams(10));
        float dist2 = dist_threshold * dist_threshold;
        for (size_t i = 0; i < knnsearch_result.size(); ++i) {
            if (out_dists_sq[i] > dist2) {
                return i;
            }
        }
        return knnsearch_result.size();
    }
}; // class DescriptorDB


class DeepDescManager
{
public: 
    // DeepDescManager( ) = default; // reserving data space (of std::vector) could be considered. but the descriptor is lightweight so don't care.
    DeepDescManager(){}
    ~DeepDescManager(){}

    DeepDescManager(const std::string &model_path, int numPoints=4096, int inputDim=3, int outputDim=256): 
    modulePath_(model_path), numPoints_(numPoints), inputDim_(inputDim), outputDim_(outputDim) {
        pointnetvlad_ = PointNetVlad(modulePath_, numPoints_, inputDim_);
        std::cout << "Deep model loaded from " << modulePath_ << std::endl;
    }

    // User-side API
    void setParams(const std::string &model_path, int numPoints=4096, int inputDim=3, int outputDim=256) {
        numPoints_ = numPoints;
        inputDim_ = inputDim;
        outputDim_ = outputDim;
        modulePath_ = model_path;
        pointnetvlad_ = PointNetVlad(modulePath_, numPoints_, inputDim_);
        std::cout << "Deep model loaded from " << modulePath_ << std::endl;
    }
    pcl::PointCloud<PointType> data_preprocess(pcl::PointCloud<PointType>::Ptr input);
    void makeAndSaveDeepDescAndKeys( pcl::PointCloud<pcl::PointXYZI> _scan_down );
    void remakeAndSaveDeepDescAndKeysAtIndex(pcl::PointCloud<PointType> _scan_down, int idx=-1);
    std::pair<int, float> detectLoopClosureID( void ); // int: nearest node index, float: relative yaw

public:
    // config 
    int TREE_MAKING_PERIOD_ = 30; // i.e., remaking tree frequency, to avoid non-mandatory every remaking, to save time cost / in the LeGO-LOAM integration, it is synchronized with the loop detection callback (which is 1Hz) so it means the tree is updated evrey 10 sec. But you can use the smaller value because it is enough fast ~ 5-50ms wrt N.
    int tree_making_period_conter = 0;

    // tree
    int    NUM_EXCLUDE_RECENT = 100; // simply just keyframe gap (related with loopClosureFrequency in yaml), but node position distance-based exclusion is ok. 
    int    NUM_CANDIDATES_FROM_TREE = 20; // 10 is enough. (refer the IROS 18 paper)
    int    TREE_MAKE_STEP_ = 3;

    double DeepLOOP_DIST_THRES = 1.0;
    double Multi_DeepLOOP_DIST_THRES = 1.0;
    double LOOP_MIN_SECOND_RATE_THRES = 1.0;

    // data 
    std::vector<double> polarcontexts_timestamp_; // optional.
    std::vector<Eigen::MatrixXd> polarcontexts_;
    std::vector<Eigen::MatrixXd> polarcontext_invkeys_;
    std::vector<Eigen::MatrixXd> polarcontext_vkeys_;

    KeyMat deepdesc_invkeys_mat_;
    KeyMat deepdesc_invkeys_to_search_;
    std::unique_ptr<InvKeyTree> deepdesc_tree_;

    std::string modulePath_;
    PointNetVlad pointnetvlad_;
    int numPoints_ = 4096;
    int inputDim_ = 3;
    int outputDim_ = 256;


}; // DeepDescManager

#endif //DEEPDESC_H
