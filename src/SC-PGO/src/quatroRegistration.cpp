#include "quatro/quatroRegistration.h"
#include <pcl/io/pcd_io.h>
#include <fstream>
#include "quatro/utility.h"
#include "CSF_filter.h"
#include <thread>
#include <future>

#include <mutex>

std::mutex csfMutex;
std::mutex fphfMutex;

const bool loopClosureEnableFlag = true;
const double mappingProcessInterval = 0.3;
const float scanPeriod = 0.1;
const int systemDelay = 0;
const int imuQueLength = 200;


boost::shared_ptr<PatchWork<QuatroPointType> > patchwork;

void initializeParams(ros::NodeHandle &nh,
                      double &voxel_size, double &normal_radius, double &fpfh_radius,
                      bool &estimating_scale, double &noise_bound, double &noise_bound_coeff,
                      double &gnc_factor, double &rot_cost_diff_thr, int &num_max_iter,
                      std::string &lidarType, std::string &groundSegMode, std::string &neighborSelectionMode,
                      CSFParams &csf_params) {
    nh.param<double>("/voxel_size", voxel_size, 0.3);
    nh.param<double>("/FPFH/normal_radius", normal_radius, 0.5);
    nh.param<double>("/FPFH/fpfh_radius", fpfh_radius, 0.75);

    nh.param<bool>("/Quatro/estimating_scale", estimating_scale, false);
    nh.param<double>("/Quatro/noise_bound", noise_bound, 0.25);
    nh.param<double>("/Quatro/noise_bound_coeff", noise_bound_coeff, 0.99);
    nh.param<double>("/Quatro/rotation/gnc_factor", gnc_factor, 1.39);
    nh.param<double>("/Quatro/rotation/rot_cost_diff_thr", rot_cost_diff_thr, 0.0001);
    nh.param<int>("/Quatro/rotation/num_max_iter", num_max_iter, 50);

    nh.param<std::string>("/Lidar_type", lidarType, "Velodyne-64-HDE");
    nh.param<std::string>("/ground_segmentation_mode", groundSegMode, "Patchwork");
    nh.param<std::string>("/neigbor_mode", neighborSelectionMode, "4CrossNeighbor");

    // csf params
    nh.param<bool>("/CSF/bSloopSmooth", csf_params.bSloopSmooth, false);
    nh.param<double>("/CSF/cloth_resolution", csf_params.cloth_resolution, 1.5);
    nh.param<double>("/CSF/rigidness", csf_params.rigidness, 3.0);
    nh.param<double>("/CSF/time_step", csf_params.time_step, 0.9);
    nh.param<double>("/CSF/class_threshold", csf_params.class_threshold, 0.5);
    nh.param<int>("/CSF/iterations", csf_params.iterations, 400);
}

void setParams(double noise_bound_of_each_measurement, double square_of_the_ratio_btw_noise_and_noise_bound,
               double estimating_scale, int num_max_iter, double control_parameter_for_gnc,
               double rot_cost_thr, const std::string &reg_type_name, Quatro<QuatroPointType, QuatroPointType>::Params &params) {
    params.noise_bound = noise_bound_of_each_measurement;
    params.cbar2 = square_of_the_ratio_btw_noise_and_noise_bound;
    params.estimate_scaling = estimating_scale;
    params.rotation_max_iterations = num_max_iter;
    params.rotation_gnc_factor = control_parameter_for_gnc;
    params.rotation_estimation_algorithm = Quatro<QuatroPointType, QuatroPointType>::ROTATION_ESTIMATION_ALGORITHM::GNC_TLS;
    params.rotation_cost_threshold = rot_cost_thr;
    params.reg_name = reg_type_name;

    if (reg_type_name == "Quatro") {
        params.inlier_selection_mode = Quatro<QuatroPointType, QuatroPointType>::INLIER_SELECTION_MODE::PMC_HEU;
    } else {
        params.inlier_selection_mode = Quatro<QuatroPointType, QuatroPointType>::INLIER_SELECTION_MODE::PMC_EXACT;
    }
}

void applyCSFFilterAsync(const pcl::PointCloud<QuatroPointType>::Ptr& cloud_in,
                         pcl::PointCloud<QuatroPointType>& ground_out,
                         pcl::PointCloud<QuatroPointType>::Ptr& non_ground_out,
                         const CSFParams& params) {
    std::cout << "" << std::endl;
    applyCSFFilter(cloud_in, ground_out, non_ground_out, params);
}

// void processImageProjection(const std::string& lidarType, const std::string& neighborSelectionMode, const std::string& groundSegMode,
//                             const pcl::PointCloud<QuatroPointType>::Ptr& rawCloud, pcl::PointCloud<QuatroPointType>& groundCloud, 
//                             pcl::PointCloud<QuatroPointType>::Ptr& nonGroundCloud, pcl::PointCloud<QuatroPointType>::Ptr& validSegments, 
//                             pcl::PointCloud<QuatroPointType>& invalidSegments) {
//     ImageProjection IP(lidarType, neighborSelectionMode, groundSegMode);
//     IP.segmentCloud(rawCloud);
//     IP.getGround(groundCloud);
//     IP.getValidSegments(*validSegments);
//     IP.getOutliers(invalidSegments);
// }

// void processPatchwork(const std::string& lidarType, const std::string& neighborSelectionMode, 
//                       const pcl::PointCloud<QuatroPointType>::Ptr& rawCloud, pcl::PointCloud<QuatroPointType>& groundCloud, 
//                       pcl::PointCloud<QuatroPointType>::Ptr& nonGroundCloud, pcl::PointCloud<QuatroPointType>::Ptr& validSegments, 
//                       pcl::PointCloud<QuatroPointType>& invalidSegments) {
//     boost::shared_ptr<PatchWork<QuatroPointType>> patchwork(new PatchWork<QuatroPointType>());
//     double t;
//     patchwork->estimate_ground(*rawCloud, groundCloud, *nonGroundCloud, t);
//     processImageProjection(lidarType, neighborSelectionMode, "Patchwork", nonGroundCloud, groundCloud, nonGroundCloud, validSegments, invalidSegments);
// }

void matchPointClouds(ros::NodeHandle &nh, pcl::PointCloud<QuatroPointType>::Ptr srcRaw, pcl::PointCloud<QuatroPointType>::Ptr tgtRaw, 
                      Quatro<QuatroPointType, QuatroPointType>::Params &params, 
                      const std::string &lidarType, const std::string &neighborSelectionMode, 
                      const std::string &groundSegMode, double voxel_size, double normal_radius, double fpfh_radius, 
                      Eigen::Matrix4d &output, const CSFParams &csf_params,
                      pcl::PointCloud<QuatroPointType>::Ptr ptrSrcNonground, pcl::PointCloud<QuatroPointType>::Ptr ptrTgtNonground,
                      bool filterGround) {

    output = Eigen::Matrix4d::Identity();

    pcl::PointCloud<QuatroPointType> srcGround, tgtGround;

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    if (filterGround) { // 如果需要滤除地面
        ImageProjection IPSrc(lidarType, neighborSelectionMode, groundSegMode);
        ImageProjection IPTgt(lidarType, neighborSelectionMode, groundSegMode);

        static double tTotal, tSrc, tTgt;

        // std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
        if (groundSegMode == "LeGO-LOAM") {
            std::cout << "Ground Segmentation Mode: LeGO-LOAM" << std::endl;
            IPSrc.segmentCloud(srcRaw);
            IPTgt.segmentCloud(tgtRaw);

            IPSrc.getGround(srcGround);
            IPTgt.getGround(tgtGround);

        } else if (groundSegMode == "Patchwork") {
            std::cout << "Ground Segmentation Mode: Patchwork" << std::endl;
            {
                std::lock_guard<std::mutex> lock(csfMutex);
                patchwork.reset(new PatchWork<QuatroPointType>(&nh));
                patchwork->estimate_ground(*(srcRaw), srcGround, *ptrSrcNonground, tSrc);
                patchwork->estimate_ground(*(tgtRaw), tgtGround, *ptrTgtNonground, tTgt);
                std::cout << "srcGround size: " << srcGround.size() << std::endl;
            }
        } else if (groundSegMode == "CSF") {
            std::cout << "Ground Segmentation Mode: CSF" << std::endl;
            {
                std::lock_guard<std::mutex> lock(csfMutex);
                applyCSFFilterAsync(srcRaw, srcGround, ptrSrcNonground, csf_params);
                applyCSFFilterAsync(tgtRaw, tgtGround, ptrTgtNonground, csf_params);
            }
        }
    } else { // 如果不需要滤除地面，直接使用原始点云
        *ptrSrcNonground = *srcRaw;
        *ptrTgtNonground = *tgtRaw;
    }

    std::chrono::system_clock::time_point after_fliter_ground = std::chrono::system_clock::now();

    pcl::PointCloud<QuatroPointType>::Ptr srcFeat(new pcl::PointCloud<QuatroPointType>);
    pcl::PointCloud<QuatroPointType>::Ptr tgtFeat(new pcl::PointCloud<QuatroPointType>);

    voxelize(ptrSrcNonground, srcFeat, voxel_size);
    voxelize(ptrTgtNonground, tgtFeat, voxel_size);

    // std::cout << "srcRaw_size: " << srcRaw->size() << std::endl;
    // std::cout << "ptrSrcNonground_size: " << ptrSrcNonground->size() << std::endl;

    if (srcFeat->empty() || tgtFeat->empty()) {
        std::cout << "error: srcFeat or tgtFeat is empty!" << std::endl;
        return;
    }

    std::chrono::system_clock::time_point before_fphf = std::chrono::system_clock::now();

    FPFHManager fpfhmanager(normal_radius, fpfh_radius);
    {
        std::lock_guard<std::mutex> lock(fphfMutex);
        fpfhmanager.flushAllFeatures();
        fpfhmanager.setFeaturePair(srcFeat, tgtFeat);
    }
    
    // std::cout << "Feature pair set: srcFeat size: " << srcFeat->size() << ", tgtFeat size: " << tgtFeat->size() << std::endl;

    // std::cout << "start matching..." << std::endl;
    pcl::PointCloud<QuatroPointType>::Ptr srcMatched(new pcl::PointCloud<QuatroPointType>);
    pcl::PointCloud<QuatroPointType>::Ptr tgtMatched(new pcl::PointCloud<QuatroPointType>);
    *srcMatched = fpfhmanager.getSrcKps();
    *tgtMatched = fpfhmanager.getTgtKps();

    std::cout << "-----srcMatched size: " << srcMatched->size() << std::endl;
    std::cout << "-----tgtMatched size: " << tgtMatched->size() << std::endl;

    if (srcMatched->size() < 6 || tgtMatched->size() < 6) {
        std::cout << "error: Not enough matched features!" << std::endl;
        return;
    }

    std::chrono::system_clock::time_point before_optim = std::chrono::system_clock::now();
    Quatro<QuatroPointType, QuatroPointType> quatro;
    quatro.setInputSource(srcMatched);
    quatro.setInputTarget(tgtMatched);
    quatro.computeTransformation(output);
    // std::cout << "Transformation matrix: " << std::endl << output << std::endl;

    std::chrono::duration<double> sec = std::chrono::system_clock::now() - start;
    std::chrono::duration<double> fliter_groung_sec = after_fliter_ground - start;
    std::chrono::duration<double> fphf_sec = before_optim - before_fphf;
    std::chrono::duration<double> optim_sec = std::chrono::system_clock::now() - before_optim;
    // std::cout << std::setprecision(4) << "\033[1;32mTotal takes: " << sec.count() << " sec. \n";
    std::cout << "fliter_groung: " << fliter_groung_sec.count() << " sec. \n";
    // std::cout << "Feature extraction: " << fphf_sec.count() << " sec. \n";
    // std::cout << "matching: " << optim_sec.count() << " sec. \033[0m" << std::endl;

    // std::cout << "Setting matching pairs: " << sec.count() - optim_sec.count() << " sec. + Quatro: " << optim_sec.count() << " sec.)\033[0m" << std::endl;
}