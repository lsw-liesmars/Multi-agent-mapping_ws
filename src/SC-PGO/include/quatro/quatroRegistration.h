#ifndef QUATRO_MATCHING_H
#define QUATRO_MATCHING_H

#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include "quatro/fpfh_manager.hpp"
#include "quatro/quatro.hpp"
#include "quatro/imageProjection.hpp"
#include "quatro/patchwork.hpp"
#include "CSF_filter.h"

extern const bool loopClosureEnableFlag;
extern const double mappingProcessInterval;
extern const float scanPeriod;
extern const int systemDelay;
extern const int imuQueLength;

extern boost::shared_ptr<PatchWork<QuatroPointType>> patchwork;

void initializeParams(ros::NodeHandle &nh,
                      double &voxel_size, double &normal_radius, double &fpfh_radius,
                      bool &estimating_scale, double &noise_bound, double &noise_bound_coeff,
                      double &gnc_factor, double &rot_cost_diff_thr, int &num_max_iter,
                      std::string &lidarType, std::string &groundSegMode, std::string &neighborSelectionMode,
                      CSFParams &csf_params);

void setParams(double noise_bound_of_each_measurement, double square_of_the_ratio_btw_noise_and_noise_bound,
               double estimating_scale, int num_max_iter, double control_parameter_for_gnc,
               double rot_cost_thr, const std::string &reg_type_name, Quatro<QuatroPointType, QuatroPointType>::Params &params);

void matchPointClouds(ros::NodeHandle &nh, pcl::PointCloud<QuatroPointType>::Ptr srcRaw, pcl::PointCloud<QuatroPointType>::Ptr tgtRaw, 
                      Quatro<QuatroPointType, QuatroPointType>::Params &params, 
                      const std::string &lidarType, const std::string &neighborSelectionMode, 
                      const std::string &groundSegMode, double voxel_size, double normal_radius, double fpfh_radius, 
                      Eigen::Matrix4d &output, const CSFParams &csf_params,
                      pcl::PointCloud<QuatroPointType>::Ptr ptrSrcNonground, pcl::PointCloud<QuatroPointType>::Ptr ptrTgtNonground,
                      bool filterGround);

#endif // QUATRO_REGISTRATION_H