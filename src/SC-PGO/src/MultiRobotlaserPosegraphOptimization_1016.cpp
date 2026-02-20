#include <fstream>
#include <math.h>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>
#include <optional>
#include <memory>
#include <unordered_map>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/registration/icp.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/visualization/pcl_visualizer.h> // 可选
#include <Eigen/Dense>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>

#include <Eigen/Dense>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"

#include "deepdesc/deepdesc.h"
#include "scancontext/Scancontext.h"
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include "scancontext/terminator.h"
#include "quatro/quatroRegistration.h"
#include "CSF_filter.h"

#include "pgo_utils.h"

using namespace gtsam;
using std::cout;
using std::endl;

# define ROBOT_NUM 3

class Robot {
public:
    double keyframeMeterGap;
    double keyframeDegGap, keyframeRadGap;
    double translationAccumulated;
    double rotationAccumulated;
    bool addAllframes = true;

    bool isNowKeyFrame;

    Pose6D odom_pose_prev;
    Pose6D odom_pose_curr;

    std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
    std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
    std::queue<sensor_msgs::NavSatFix::ConstPtr> gpsBuf;
    std::queue<std::pair<int, int>> initLoopICPBuf;
    std::queue<std::tuple<int, int, int, int>> interRobotLoopICPBuf;

    std::unique_ptr<std::mutex> mBuf;

    ros::Time timeLaserOdometry;
    ros::Time timeLaser;

    Terminator terminator;
    std::unique_ptr<std::atomic<bool>> shutdown;

    pcl::PointCloud<PointType>::Ptr laserCloudFullRes;
    pcl::PointCloud<PointType>::Ptr laserCloudMapAfterPGO;

    std::unique_ptr<std::mutex> mKF;
    std::vector<pcl::PointCloud<PointType>::Ptr> keyframeLaserClouds;
    std::vector<pcl::PointCloud<PointType>::Ptr> allframeLaserClouds;  
    std::vector<Pose6D> keyframePoses;
    std::vector<Pose6D> keyframePosesUpdated;
    std::vector<Pose6D> allframePoses;
    std::unordered_map<int, int> keyframeIndexTOallID; 
    std::vector<Pose6D> DenseframePoses;
    std::vector<ros::Time> keyframeTimes;
    std::vector<ros::Time> DenseframeTimes;

    std::unique_ptr<std::atomic<bool>> hasNewScanForLC;

    std::unique_ptr<std::mutex> loopPairMutex;
    std::unique_ptr<std::mutex> interloopPairMutex;
    std::unordered_map<int, int> loopIndexContainer;
    std::unordered_map<int, std::pair<int, int>> interRobotLoopIndexContainer;

    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtreeHistoryKeyPoses;
    std::unique_ptr<std::atomic<bool>> gtSAMgraphMade;

    std::unique_ptr<std::mutex> mtxPosegraph;
    std::shared_ptr<std::mutex> mtxPosegraph_dense;
    std::shared_ptr<std::mutex> mtxPosegraph_isam;
    gtsam::NonlinearFactorGraph* gtSAMgraph;
    gtsam::Values initialEstimate;
    std::vector<ros::Time> newStateTimes;
    std::vector<ros::Time> isamStateTimes;
    gtsam::ISAM2 *isam;
    gtsam::Values isamCurrentEstimate;
    int numOptimizeFrames;

    noiseModel::Diagonal::shared_ptr priorNoise;
    noiseModel::Diagonal::shared_ptr odomNoise;
    noiseModel::Base::shared_ptr robustLoopNoise;
    noiseModel::Base::shared_ptr initTw_Noise;
    noiseModel::Base::shared_ptr robustGPSNoise;

    pcl::VoxelGrid<PointType> downSizeFilterLoopClosure;
    SCManager scManager;
    DeepDescManager deepDescManager;
    std::unique_ptr<std::mutex> scMutex;
    std::unique_ptr<std::mutex> DeepDescMutex;

    std::unique_ptr<std::mutex> downICPFilterMutex;
    double scDistThres, scMaximumRadius, Multi_scDistThres;

    pcl::VoxelGrid<PointType> downSizeFilterICP;

    pcl::PointCloud<PointType>::Ptr laserCloudMapPGO;
    pcl::VoxelGrid<PointType> downSizeFilterMapPGO;
    bool laserCloudMapPGORedraw;

    bool useGPS;
    sensor_msgs::NavSatFix::ConstPtr currGPS;
    bool hasGPSforThisKF;
    bool gpsOffsetInitialized;
    double gpsAltitudeInitOffset;

    std::unique_ptr<std::mutex> mtxRecentPose;
    double recentOptimizedX;
    double recentOptimizedY;
    std::unique_ptr<std::atomic<int>> recentIdxUpdated;
    Eigen::Affine3f W_T_odom;

    ros::Publisher pubMapAftPGO, pubOdomAftPGO, pubPathAftPGO, pubPathOdom;
    ros::Publisher pubLoopScanLocal, pubLoopSubmapLocal;
    ros::Publisher pubOdomRepubVerifier;
    ros::Publisher pubKeyFramesId;
    ros::Publisher pubLoopScanLocalRegisted;
    ros::Publisher pub_multiLoopScanLocalRegisted;
    ros::Publisher pubLoopConstraintEdge;
    ros::Publisher pub_multiLoopConstraintEdge;

    std::string save_directory;
    std::string deepmodelpath;
    std::string pgTUMFormat, pgScansDirectory;
    std::string saveSCDDirectory;
    std::string saveSubmapDirectory;
    std::string odomTUMFormat;
    std::string denseTUMFormat;
    std::string pgKITTIformat;
    std::string odomKITTIformat;

    std::ofstream scanMatchStream;
    std::ofstream robots_scanMatchStream;
    std::fstream pgSaveStream; // pg: pose-graph
    std::fstream pgTimeSaveStream;

    int lcd_mode = 0; // 0:radiuSearch, 1: SC search, 2: DeepDesc search
    int halfsubmapsize = 5;
    int halfsubmapsize_icp = 5;
    double last_frame_sub_time;

    double historyKeyframeSearchRadius;
    double historyKeyframeSearchTimeDiff;
    int historyKeyframeSearchNum;
    double loopClosureFrequency;
    int graphUpdateTimes;
    double graphUpdateFrequency;
    double loopNoiseScore;
    double vizmapFrequency;
    double vizPathFrequency;
    double speedFactor;
    double filter_size;
    double icp_filter_size;
    double loopFitnessScoreThreshold;
    double Multi_loopFitnessScoreThreshold;
    // 定义重建树的周期
    int TREE_MAKING_PERIOD = 30;
    int tree_making_counter = 0;
    int last_inter_lcd_size = 0;

    int robot_id;
    std::unordered_map<int, std::shared_ptr<Robot>> friend_robots;
    int index_offset;

    Eigen::Affine3f scInitialGuess;
    std::mutex scGuessMutex;

    // Quatro
    bool useQuatro;
    double voxel_size, normal_radius, fpfh_radius;
    bool estimating_scale;
    double noise_bound, noise_bound_coeff, gnc_factor, rot_cost_diff_thr;
    int num_max_iter;
    std::string lidarType, groundSegMode, neighborSelectionMode;
    std::mutex matchMutex;

    Quatro<PointType, PointType>::Params quatroParams;
    CSFParams csf_params;

    // void initializeQuatroParams(ros::NodeHandle &nh);

    Robot(ros::NodeHandle &nh, int id, gtsam::ISAM2 *isam_global, std::shared_ptr<std::mutex> mtxPosegraph_global) :
        robot_id(id), 
        // robots(robots_ref),

        keyframeMeterGap(2.0), keyframeDegGap(10.0), // keyframeRadGap(deg2rad(keyframeDegGap)),
        translationAccumulated(1000000.0), rotationAccumulated(1000000.0),
        isNowKeyFrame(false), odom_pose_prev({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}), odom_pose_curr({0.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
        timeLaserOdometry(0, 0), timeLaser(0, 0), hasNewScanForLC(std::make_unique<std::atomic<bool>>(false)), gtSAMgraphMade(std::make_unique<std::atomic<bool>>(false)),
        laserCloudFullRes(new pcl::PointCloud<PointType>()), laserCloudMapAfterPGO(new pcl::PointCloud<PointType>()),
        laserCloudMapPGO(new pcl::PointCloud<PointType>()), laserCloudMapPGORedraw(true),
        useGPS(true), hasGPSforThisKF(false), gpsOffsetInitialized(false), gpsAltitudeInitOffset(0.0),
        recentOptimizedX(0.0), recentOptimizedY(0.0), recentIdxUpdated(std::make_unique<std::atomic<int>>(0)),
        isam(nullptr), terminator(300, 10.0),
        mBuf(std::make_unique<std::mutex>()), mKF(std::make_unique<std::mutex>()), loopPairMutex(std::make_unique<std::mutex>()), interloopPairMutex(std::make_unique<std::mutex>()),
        mtxPosegraph(std::make_unique<std::mutex>()), mtxRecentPose(std::make_unique<std::mutex>()),mtxPosegraph_dense(std::make_unique<std::mutex>()),
        scMutex(std::make_unique<std::mutex>()), downICPFilterMutex(std::make_unique<std::mutex>()), 
        DeepDescMutex(std::make_unique<std::mutex>()),
        shutdown(std::make_unique<std::atomic<bool>>(false)),
        kdtreeHistoryKeyPoses(new pcl::KdTreeFLANN<pcl::PointXYZ>())
    {
        addAllframes = false;
        W_T_odom = Eigen::Affine3f::Identity();
        pubMapAftPGO = nh.advertise<sensor_msgs::PointCloud2>("/aft_pgo_map_" + std::to_string(id), 100);
        pubOdomAftPGO = nh.advertise<nav_msgs::Odometry>("/aft_pgo_odom_" + std::to_string(id), 100);
        pubPathAftPGO = nh.advertise<nav_msgs::Path>("/aft_pgo_path_" + std::to_string(id), 100);
        pubPathOdom = nh.advertise<nav_msgs::Path>("/odom_path_" + std::to_string(id), 100);
        pubLoopScanLocal = nh.advertise<sensor_msgs::PointCloud2>("/loop_scan_local_" + std::to_string(id), 100);
        pubLoopSubmapLocal = nh.advertise<sensor_msgs::PointCloud2>("/loop_submap_local_" + std::to_string(id), 100);
        pubOdomRepubVerifier = nh.advertise<nav_msgs::Odometry>("/repub_odom_" + std::to_string(id), 100);
        pubKeyFramesId = nh.advertise<std_msgs::Header>("/key_frames_ids_" + std::to_string(id), 10);
        pubLoopScanLocalRegisted = nh.advertise<sensor_msgs::PointCloud2>("/loop_scan_local_registed_" + std::to_string(id), 100);
        pub_multiLoopScanLocalRegisted = nh.advertise<sensor_msgs::PointCloud2>("/multi_loop_scan_local_registed_" + std::to_string(id), 100);
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/loop_closure_constraints_" + std::to_string(id), 1);
        pub_multiLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/multi_loop_closure_constraints_" + std::to_string(id), 1);
        
        gtSAMgraph = new gtsam::NonlinearFactorGraph();
        isam = isam_global;
        mtxPosegraph_isam = mtxPosegraph_global;
        index_offset = robot_id * MAX_NODES_PER_ROBOT;
        init_params(nh, id);
        numOptimizeFrames = 0;
        initializeQuatroParams(nh);
        keyframeRadGap = keyframeDegGap * M_PI / 180.0;
    }

    void initializeQuatroParams(ros::NodeHandle &nh) 
    {
        nh.param<bool>("use_quatro", useQuatro, false);
        if (!useQuatro) {
            return;
        }
        try {
        initializeParams(nh, voxel_size, normal_radius, fpfh_radius, estimating_scale, noise_bound, noise_bound_coeff,
                         gnc_factor, rot_cost_diff_thr, num_max_iter, lidarType, groundSegMode, neighborSelectionMode, csf_params);

        setParams(noise_bound, noise_bound_coeff, estimating_scale, num_max_iter, gnc_factor, rot_cost_diff_thr, "Quatro", quatroParams);
        } catch (const std::exception &e) {
            std::cerr << "Exception in initializeQuatroParams: " << e.what() << std::endl;
        }
    }

    void init_params(ros::NodeHandle &nh, int id)
    {
        bool terminateAtEnd = false;
        nh.param<bool>("terminate_at_end", terminateAtEnd, false);
        nh.param<bool>("use_gps", useGPS, true);
        if (!terminateAtEnd)
        {
            terminator.setWaitPacketsForNextPacket(-1);
        }
        nh.param<std::string>("save_directory", save_directory, "/");
        save_directory += "/robot_" + std::to_string(id) + "/";
        auto unused = system((std::string("mkdir -p ") + save_directory).c_str());

        pgTUMFormat = save_directory + "keyscan_optimized_poses.txt";
        odomTUMFormat = save_directory + "keyscan_odom_poses.txt";
        denseTUMFormat = save_directory + "dense_optimized_poses.txt";

        pgSaveStream = std::fstream(save_directory + "singlesession_posegraph.g2o", std::fstream::out);
        pgSaveStream.precision(std::numeric_limits<double>::max_digits10);

        pgTimeSaveStream = std::fstream(save_directory + "keyscan_times.txt", std::fstream::out);
        pgTimeSaveStream.precision(std::numeric_limits<double>::max_digits10);
        scanMatchStream = std::ofstream(save_directory + "keyscan_matches.txt", std::fstream::out);
        robots_scanMatchStream = std::ofstream(save_directory + "robots_scanMatchStream.txt", std::fstream::out);
        pgScansDirectory = save_directory + "Scans/";
        unused = system((std::string("mkdir -p ") + pgScansDirectory).c_str());
        unused = system((std::string("exec rm -r ") + pgScansDirectory).c_str());
        unused = system((std::string("mkdir -p ") + pgScansDirectory).c_str());

        saveSCDDirectory = save_directory + "SCDs/";
        unused = system((std::string("mkdir -p ") + saveSCDDirectory).c_str());
        unused = system((std::string("exec rm -r ") + saveSCDDirectory).c_str());
        unused = system((std::string("mkdir -p ") + saveSCDDirectory).c_str());

        saveSubmapDirectory = save_directory + "Submaps/";
        unused = system((std::string("mkdir -p ") + saveSubmapDirectory).c_str());
        unused = system((std::string("exec rm -r ") + saveSubmapDirectory).c_str());
        unused = system((std::string("mkdir -p ") + saveSubmapDirectory).c_str());

        nh.param<double>("keyframe_meter_gap", keyframeMeterGap, 1.0);
        nh.param<double>("keyframe_deg_gap", keyframeDegGap, 2.0);
        // keyframeRadGap = deg2rad(keyframeDegGap);
        keyframeRadGap = keyframeDegGap * M_PI / 180.0;


        // loop closure detection parameter setting
        nh.param<int>("lcd_mode", lcd_mode, 0);
        nh.param<int>("halfsubmapsize", halfsubmapsize, 5);
        nh.param<int>("halfsubmapsize_icp", halfsubmapsize_icp, 5);
        nh.param<std::string>("deepmodelpath", deepmodelpath, "/");
        last_frame_sub_time = ros::Time::now().toSec();

        if(lcd_mode==1){
            nh.param<double>("sc_dist_thres", scDistThres, 0.2);
            nh.param<double>("Multi_Agents_SC_DIST_THRES", Multi_scDistThres, 0.2);
            nh.param<double>("sc_max_radius", scMaximumRadius, 80.0);
        }
        else if(lcd_mode==2){ // 
            // std::string deepmodelpath = "/home/wbl/code/kuangshan/deep_reloc/PointNetVlad-best-1018.pt";
            deepDescManager.setParams(deepmodelpath, 4096, 3, 256); // std::unique_ptr cannot use "=" to copy
            nh.param<int>("num_exclude_recent", deepDescManager.NUM_EXCLUDE_RECENT, 100);
            nh.param<double>("loop_desc_dist_thres", deepDescManager.DeepLOOP_DIST_THRES, 0.1);
            nh.param<double>("Multi_Agents_loop_desc_dist_thres", deepDescManager.Multi_DeepLOOP_DIST_THRES, 0.1);  
            nh.param<double>("loop_desc_min_second_rate_thres", deepDescManager.LOOP_MIN_SECOND_RATE_THRES, 0.5);
            nh.param<int>("tree_make_step", deepDescManager.TREE_MAKE_STEP_, 1);
        }

        // nh.param<bool>("radius_search", radiusSearch, false);
        if (lcd_mode==0)
            ROS_INFO_STREAM("Using radius search for loop closure.");
        else if (lcd_mode==1)
            ROS_INFO_STREAM("Using scan context for loop closure.");
        else if (lcd_mode==2)
            ROS_INFO_STREAM("Using deepdesc for loop closure.");
        else
            ROS_INFO_STREAM("Invalid loop closure detection mode.");

        nh.param<double>("historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0);
        nh.param<double>("historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 10.0);
        nh.param<double>("loopNoiseScore", loopNoiseScore, 0.5);
        nh.param<int>("graphUpdateTimes", graphUpdateTimes, 2);
        nh.param<double>("loopFitnessScoreThreshold", loopFitnessScoreThreshold, 1.0);
        nh.param<double>("InterloopFitnessScoreThreshold", Multi_loopFitnessScoreThreshold, 3.0);
        nh.param<int>("historyKeyframeSearchNum", historyKeyframeSearchNum, 3);

        nh.param<double>("speedFactor", speedFactor, 1);
        nh.param<double>("lcd_filter_size", filter_size, 0.4);
        nh.param<double>("icp_filter_size", icp_filter_size, 0.4);
        nh.param<double>("loopClosureFrequency", loopClosureFrequency, 2);
        loopClosureFrequency *= speedFactor;
        nh.param<double>("graphUpdateFrequency", graphUpdateFrequency, 1.0);
        graphUpdateFrequency *= speedFactor;
        nh.param<double>("vizmapFrequency", vizmapFrequency, 10);
        vizmapFrequency *= speedFactor;
        nh.param<double>("vizPathFrequency", vizPathFrequency, 10);
        vizPathFrequency *= speedFactor;

        initNoises(loopNoiseScore);

        scManager.setSCdistThres(scDistThres);
        scManager.setMaximumRadius(scMaximumRadius);

        downSizeFilterLoopClosure.setLeafSize(filter_size, filter_size, filter_size);
        downSizeFilterICP.setLeafSize(icp_filter_size, icp_filter_size, icp_filter_size);

        double mapVizFilterSize;
        nh.param<double>("mapviz_filter_size", mapVizFilterSize, 0.4);
        downSizeFilterMapPGO.setLeafSize(mapVizFilterSize, mapVizFilterSize, mapVizFilterSize);
    }

    void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &_laserOdometry)
    {
        // std::cout<< "odometryBuf.size(): " << odometryBuf.size() << std::endl;
        std::lock_guard<std::mutex> lock(*mBuf);
        odometryBuf.push(_laserOdometry);
        terminator.newPacket();
    }

    void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &_laserCloudFullRes)
    {
        std::lock_guard<std::mutex> lock(*mBuf);
        fullResBuf.push(_laserCloudFullRes);
    }

    void gpsHandler(const sensor_msgs::NavSatFix::ConstPtr &_gps)
    {
        if (useGPS) {
            std::lock_guard<std::mutex> lock(*mBuf);
            gpsBuf.push(_gps);
        }
    }

    void initNoises(double loopNoiseScore)
    {
        gtsam::Vector priorNoiseVector6(6);
        priorNoiseVector6 << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12;
        priorNoise = noiseModel::Diagonal::Variances(priorNoiseVector6);

        gtsam::Vector odomNoiseVector6(6);
        odomNoiseVector6 << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
        odomNoise = noiseModel::Diagonal::Variances(odomNoiseVector6);

        gtsam::Vector robustNoiseVector6(6);
        robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore;
        robustLoopNoise = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Cauchy::Create(1),
            gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6));
        // robustLoopNoise = noiseModel::Diagonal::Variances(robustNoiseVector6);

        gtsam::Vector initTw_NoiseVector6(6);
        initTw_NoiseVector6 << 10000, 10000, 10000, 10000, 10000, 10000;
        initTw_Noise = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Cauchy::Create(1),
            gtsam::noiseModel::Diagonal::Variances(initTw_NoiseVector6));

        double bigNoiseTolerentToXY = 1000000000.0; // 1e9
        double gpsAltitudeNoiseScore = 250.0;
        gtsam::Vector robustNoiseVector3(3);
        robustNoiseVector3 << bigNoiseTolerentToXY, bigNoiseTolerentToXY, gpsAltitudeNoiseScore;
        robustGPSNoise = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Cauchy::Create(1),
            gtsam::noiseModel::Diagonal::Variances(robustNoiseVector3));
    }

    void pubPath()
    {
        nav_msgs::Odometry odomAftPGO;
        nav_msgs::Path pathAftPGO;
        pathAftPGO.header.frame_id = "camera_init";
        // std::cout<<"robot_id: "<<robot_id<<", recentIdxUpdated: "<<*recentIdxUpdated<<std::endl;
        mKF->lock();
        for (int node_idx = 0; node_idx < *recentIdxUpdated; node_idx++)
        {
            const Pose6D &pose_est = keyframePosesUpdated.at(node_idx);

            nav_msgs::Odometry odomAftPGOthis;
            odomAftPGOthis.header.frame_id = "camera_init";
            odomAftPGOthis.child_frame_id = "/aft_pgo";
            odomAftPGOthis.header.stamp = keyframeTimes.at(node_idx);
            odomAftPGOthis.pose.pose.position.x = pose_est.x;
            odomAftPGOthis.pose.pose.position.y = pose_est.y;
            odomAftPGOthis.pose.pose.position.z = pose_est.z;
            odomAftPGOthis.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(pose_est.roll, pose_est.pitch, pose_est.yaw);
            odomAftPGO = odomAftPGOthis;

            geometry_msgs::PoseStamped poseStampAftPGO;
            poseStampAftPGO.header = odomAftPGOthis.header;
            poseStampAftPGO.pose = odomAftPGOthis.pose.pose;

            pathAftPGO.header.stamp = odomAftPGOthis.header.stamp;
            pathAftPGO.header.frame_id = "camera_init";
            pathAftPGO.poses.push_back(poseStampAftPGO);
        }

        // pub odom path
        nav_msgs::Path pathOdom;
        pathOdom.header.frame_id = "camera_init";
        for (int node_idx = 0; node_idx < int(keyframePoses.size()); node_idx++)
        {
            const Pose6D &pose_est = keyframePoses.at(node_idx);

            nav_msgs::Odometry odomAftPGOthis;
            odomAftPGOthis.header.frame_id = "camera_init";
            odomAftPGOthis.child_frame_id = "/aft_pgo";
            odomAftPGOthis.header.stamp = keyframeTimes.at(node_idx);
            odomAftPGOthis.pose.pose.position.x = pose_est.x;
            odomAftPGOthis.pose.pose.position.y = pose_est.y;
            odomAftPGOthis.pose.pose.position.z = pose_est.z;
            odomAftPGOthis.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(pose_est.roll, pose_est.pitch, pose_est.yaw);
            odomAftPGO = odomAftPGOthis;

            geometry_msgs::PoseStamped poseStampAftPGO;
            poseStampAftPGO.header = odomAftPGOthis.header;
            poseStampAftPGO.pose = odomAftPGOthis.pose.pose;

            pathOdom.header.stamp = odomAftPGOthis.header.stamp;
            pathOdom.header.frame_id = "camera_init";
            pathOdom.poses.push_back(poseStampAftPGO);
        }
        mKF->unlock();
        pubOdomAftPGO.publish(odomAftPGO);
        pubPathAftPGO.publish(pathAftPGO);
        pubPathOdom.publish(pathOdom);

        static tf::TransformBroadcaster br;
        tf::Transform transform;
        tf::Quaternion q;
        transform.setOrigin(tf::Vector3(odomAftPGO.pose.pose.position.x, odomAftPGO.pose.pose.position.y, odomAftPGO.pose.pose.position.z));
        q.setW(odomAftPGO.pose.pose.orientation.w);
        q.setX(odomAftPGO.pose.pose.orientation.x);
        q.setY(odomAftPGO.pose.pose.orientation.y);
        q.setZ(odomAftPGO.pose.pose.orientation.z);
        transform.setRotation(q);
    }

    void updatePoses()
    {
        mKF->lock();
        int numPosesUpdated = numOptimizeFrames;
        // std::cout<<"robot_id: "<<robot_id<<", numPosesUpdated: "<<numPosesUpdated<<std::endl;
        for (int node_idx = 0; node_idx < numPosesUpdated; node_idx++)
        {
            // find key=index_offset+node_idx in isamCurrentEstimate, if not found, continue
            if (!isamCurrentEstimate.exists(index_offset+node_idx))
                continue;
            // std::cout<<"robot_id: "<<robot_id<<", node_idx: "<<node_idx<<std::endl;
            Pose6D &p = keyframePosesUpdated[node_idx];
            p.x = isamCurrentEstimate.at<gtsam::Pose3>(index_offset+node_idx).translation().x();
            p.y = isamCurrentEstimate.at<gtsam::Pose3>(index_offset+node_idx).translation().y();
            p.z = isamCurrentEstimate.at<gtsam::Pose3>(index_offset+node_idx).translation().z();
            p.roll = isamCurrentEstimate.at<gtsam::Pose3>(index_offset+node_idx).rotation().roll();
            p.pitch = isamCurrentEstimate.at<gtsam::Pose3>(index_offset+node_idx).rotation().pitch();
            p.yaw = isamCurrentEstimate.at<gtsam::Pose3>(index_offset+node_idx).rotation().yaw();
            // std::cout<< "keyframePosesUpdated[" << node_idx << "]: " << p.x << ", " << p.y << ", " << p.z << ", " << p.roll << ", " << p.pitch << ", " << p.yaw << std::endl;
            // std::cout<< "keyframePoses[" << node_idx << "]: " << keyframePoses[node_idx].x << ", " << keyframePoses[node_idx].y << ", " << keyframePoses[node_idx].z << ", " << keyframePoses[node_idx].roll << ", " << keyframePoses[node_idx].pitch << ", " << keyframePoses[node_idx].yaw << std::endl;
        }
        mKF->unlock();

        mtxRecentPose->lock();
        // const gtsam::Pose3 &lastOptimizedPose = isamCurrentEstimate.at<gtsam::Pose3>(index_offset+int(numPosesUpdated) - 1);
        // recentOptimizedX = lastOptimizedPose.translation().x();
        // recentOptimizedY = lastOptimizedPose.translation().y();
        auto lastOptimizedPose = keyframePosesUpdated[numPosesUpdated - 1];
        recentOptimizedX = lastOptimizedPose.x;
        recentOptimizedY = lastOptimizedPose.y;
        *recentIdxUpdated = numPosesUpdated - 1;
        W_T_odom = toEigenAffine3f(lastOptimizedPose) * toEigenAffine3f(keyframePoses[numPosesUpdated - 1]).inverse();
        // std::cout<<"robot_id: "<<robot_id<<", numPosesUpdated: "<<numPosesUpdated<<", recentIdxUpdated: "<<*recentIdxUpdated<<std::endl;

        mtxRecentPose->unlock();
    }

    void runISAM2opt()
    {
        mtxPosegraph_isam->lock();
        mtxPosegraph->lock();
        if(!isam->calculateEstimate().exists(0) && robot_id!=0){
            mtxPosegraph->unlock();
            mtxPosegraph_isam->unlock();
            return;
        }
        if (newStateTimes.size() != initialEstimate.size()) {
            ROS_ERROR("Inconsistent new state times and initial estimates.");
        }
        isamStateTimes.insert(isamStateTimes.end(), newStateTimes.begin(), newStateTimes.end());
        // std::cout<<"here1 robot_id: "<<robot_id<<", newStateTimes.size(): "<<newStateTimes.size()<<std::endl;
        isam->update(*gtSAMgraph, initialEstimate);
        // std::cout<<"here2 robot_id: "<<robot_id<<", isam->update()"<<std::endl;
        isam->update();
        // std::cout<<"here3 robot_id: "<<robot_id<<", isam->update()"<<std::endl;
        // std::cout << "isam->update() done" << std::endl;

        laserCloudMapPGORedraw = true;  // newStateTimes.size() > 0;
        numOptimizeFrames += newStateTimes.size();
        gtSAMgraph->resize(0);
        initialEstimate.clear();
        newStateTimes.clear();

        // std::cout<<"here11111111114 robot_id: "<<robot_id<<", isam->update()"<<std::endl;
        isamCurrentEstimate = isam->calculateEstimate();
        // std::cout<<"here11111111115 robot_id: "<<robot_id<<", isam->update()"<<std::endl;
        updatePoses();

        mtxPosegraph->unlock();
        mtxPosegraph_isam->unlock();
    }

   void loopFindNearKeyframesCloud(pcl::PointCloud<PointType>::Ptr &nearKeyframes, pcl::PointCloud<PointType>::Ptr &nearKeyframes_DS,
                                const int &key, const int &submap_size, const int &root_idx)
    {
        nearKeyframes->clear();
        nearKeyframes_DS->clear();
        // mKF->lock();
        pcl::PointCloud<PointType>::Ptr tempKeyframes(new pcl::PointCloud<PointType>());

        for (int i = -submap_size; i <= submap_size; ++i)
        {
            std::lock_guard<std::mutex> lock(*mKF);
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= int(keyframeLaserClouds.size()) || keyNear >= int(keyframePoses.size()))
                continue;
            if (keyframeLaserClouds[keyNear]->empty())
                continue;
            *tempKeyframes += *local2global(keyframeLaserClouds[keyNear], keyframePoses[keyNear]);
        }
        // mKF->unlock();

        if (tempKeyframes->empty())
            return;

        // Remove NaN points
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*tempKeyframes, *tempKeyframes, indices);

        const Pose6D W_T_B = keyframePoses[root_idx];
        Eigen::Affine3f affine_W_T_B = pcl::getTransformation(W_T_B.x, W_T_B.y, W_T_B.z, W_T_B.roll, W_T_B.pitch, W_T_B.yaw);
        gtsam::Pose3 gtsam_W_T_B(gtsam::Rot3(affine_W_T_B.rotation().cast<double>()), affine_W_T_B.translation().cast<double>());
        gtsam::Pose3 gtsam_B_T_W = gtsam_W_T_B.inverse();
        pcl::PointCloud<PointType>::Ptr transformedNearKeyframes = transformPointCloud(tempKeyframes, gtsam_B_T_W);

        // Downsample the point cloud
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        {
            // std::scoped_lock lock(*downICPFilterMutex);
            std::lock_guard<std::mutex> lock(*downICPFilterMutex);
            downSizeFilterICP.setInputCloud(transformedNearKeyframes);
            downSizeFilterICP.filter(*cloud_temp);

        }
        *nearKeyframes = *transformedNearKeyframes;
        *nearKeyframes_DS = *cloud_temp;
    }

    void loopFindNearframesCloud( pcl::PointCloud<PointType>::Ptr& local_nearframes, const int& key, const int& submap_size, const int& root_idx)
    {
        // extract and stacking near keyframes (in global coord)
        pcl::PointCloud<PointType>::Ptr nearframes(new pcl::PointCloud<PointType>());
        Eigen::Vector3d transPos = Eigen::Vector3d(0,0,0);
        transPos << allframePoses[root_idx].x, allframePoses[root_idx].y, allframePoses[root_idx].z;
        for (int i = -submap_size; i <= submap_size; ++i) {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= int(allframeLaserClouds.size()) ){
                // std::cout << "keyNear: err!! " <<keyNear<< std::endl; 
                continue;
            }

            // odometry T_B1_B2
            // *nearKeyframes += * local2global(keyframeLaserClouds[keyNear], keyframePosesUpdated[root_idx]);
            // TODO binliang: transPos ??
            *nearframes += * local2global(allframeLaserClouds[keyNear], allframePoses[keyNear], transPos); // to global
            // *nearKeyframes += * global2local(local2global(keyframeLaserClouds[keyNear], keyframePosesUpdated[keyNear]), 
            //                                                                     keyframePosesUpdated[key]); // to local
        }

        // add by yiwen
        const Pose6D W_T_B = allframePoses[root_idx];
        Eigen::Affine3f affine_W_T_B = pcl::getTransformation(W_T_B.x, W_T_B.y, W_T_B.z, W_T_B.roll, W_T_B.pitch, W_T_B.yaw);
        gtsam::Pose3 gtsam_W_T_B(gtsam::Rot3(affine_W_T_B.rotation().cast<double>()), affine_W_T_B.translation().cast<double>());
        gtsam::Pose3 gtsam_B_T_W = gtsam_W_T_B.inverse();
        local_nearframes = transformPointCloud(nearframes, gtsam_B_T_W);
    }

    std::optional<gtsam::Pose3> doICPVirtualRelative(int _loop_kf_idx, int _curr_kf_idx)
    {
        pcl::PointCloud<PointType>::Ptr currKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud_DS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr targetKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr targetKeyframeCloud_DS(new pcl::PointCloud<PointType>());


        loopFindNearKeyframesCloud(currKeyframeCloud, cureKeyframeCloud_DS, _curr_kf_idx, historyKeyframeSearchNum, _curr_kf_idx);
        loopFindNearKeyframesCloud(targetKeyframeCloud, targetKeyframeCloud_DS, _loop_kf_idx, historyKeyframeSearchNum, _loop_kf_idx);

        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(2 * historyKeyframeSearchNum * keyframeMeterGap);
        icp.setMaximumIterations(50);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-1);

        icp.setInputSource(cureKeyframeCloud_DS);
        icp.setInputTarget(targetKeyframeCloud_DS);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        mKF->lock();
        // Pose6D loopKfPose = keyframePosesUpdated[_loop_kf_idx];
        // Eigen::Affine3f guess = toEigenAffine3f(loopKfPose).inverse() * toEigenAffine3f(keyframePosesUpdated[_curr_kf_idx]);
        Pose6D loopKfPose = keyframePoses[_loop_kf_idx];
        Eigen::Affine3f guess = toEigenAffine3f(loopKfPose).inverse() * toEigenAffine3f(keyframePoses[_curr_kf_idx]);
        mKF->unlock();
        icp.align(*unused_result, guess.matrix());

        float rotang = diffRotation(guess.rotation(), icp.getFinalTransformation().block<3, 3>(0, 0));

        if (icp.hasConverged() == false || icp.getFitnessScore() > loopFitnessScoreThreshold)
        {
            // ROS_INFO_STREAM("[SC loop] ICP fitness test failed (" << icp.getFitnessScore() << " > " << loopFitnessScoreThreshold << "). Reject this SC loop.");
            return std::nullopt;
        }
        else
        {
            // ROS_INFO_STREAM("[SC loop] ICP fitness test passed (" << icp.getFitnessScore() << " < " << loopFitnessScoreThreshold
                                                                                    //    << ", rot angle change " << rotang * 180 / M_PI << "). Add this SC loop.");
            loopPairMutex->lock();
            loopIndexContainer[_curr_kf_idx] = _loop_kf_idx;
            loopPairMutex->unlock();

            sensor_msgs::PointCloud2 cureKeyframeCloudRegMsg;
            pcl::PointCloud<PointType>::Ptr cureKeyframeCloudWorldReg(new pcl::PointCloud<PointType>());
            cureKeyframeCloudWorldReg = local2global(unused_result, loopKfPose);
            pcl::toROSMsg(*cureKeyframeCloudWorldReg, cureKeyframeCloudRegMsg);
            cureKeyframeCloudRegMsg.header.frame_id = "camera_init";
            pubLoopScanLocalRegisted.publish(cureKeyframeCloudRegMsg);
        }

        Eigen::Affine3f correctionLidarFrame(icp.getFinalTransformation());
        Eigen::Matrix3f rot = correctionLidarFrame.rotation();
        Eigen::Vector3f trans = correctionLidarFrame.translation();

        return gtsam::Pose3(gtsam::Rot3(rot.cast<double>()), trans.cast<double>());
    }

    void logRotationAngleAndRobots(int curr_robot_id, int friend_robot_id, const Eigen::Matrix4f& initial_guess, double icp_fitness_score)
    {
        // 提取旋转矩阵 (initial_guess 的 3x3 子矩阵)
        Eigen::Matrix3f rotation_matrix = initial_guess.block<3, 3>(0, 0);

        // 将旋转矩阵转换为欧拉角（xyz顺序）
        Eigen::Vector3f euler_angles = rotation_matrix.eulerAngles(0, 1, 2); // 绕x、y、z轴的旋转角度

        // 计算旋转角度（例如绕z轴的旋转）
        float yaw_angle = euler_angles[2]; // 获取绕z轴的旋转角

        // 打开文件，准备写入（如果文件不存在会自动创建）
        std::ofstream outfile;
        outfile.open("/home/cyw/temp/rotation_log.txt", std::ios_base::app); // 以追加模式打开

        // 写入两个机器人ID和旋转角度（单位为弧度，可以将其转换为度）
        outfile << "Robot " << curr_robot_id << " and Robot " << friend_robot_id
                << " Initial Guess Rotation (yaw in radians): " << yaw_angle << 
                ", ICP Fitness Score: " << icp_fitness_score << "\n";

        // 关闭文件
        outfile.close();
    }

    std::optional<gtsam::Pose3> doICPVirtualRelativeFriendRobot(Robot* curr_robot, Robot* friend_robot, int curr_node_idx, int friend_node_idx)
    {
        if (curr_node_idx >= curr_robot->keyframePoses.size() || friend_node_idx >= friend_robot->keyframePoses.size()) {
            ROS_ERROR("Node index out of bounds");
            return std::nullopt;
        }

        pcl::PointCloud<PointType>::Ptr currKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr currKeyframeCloud_DS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr friendKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr friendKeyframeCloud_DS(new pcl::PointCloud<PointType>());

        curr_robot->loopFindNearKeyframesCloud(currKeyframeCloud, currKeyframeCloud_DS, 
                                            curr_node_idx, historyKeyframeSearchNum, curr_node_idx);
        friend_robot->loopFindNearKeyframesCloud(friendKeyframeCloud, friendKeyframeCloud_DS, 
                                                friend_node_idx, historyKeyframeSearchNum, friend_node_idx);


        int cur_global_idx = curr_node_idx + curr_robot->index_offset;
        int friend_global_idx = friend_node_idx + friend_robot->index_offset;

        // std::cout<<"cur_global_idx: "<<cur_global_idx<<", friend_global_idx: "<<friend_global_idx<<std::endl;

        // Remove NaN points from the point clouds
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*currKeyframeCloud_DS, *currKeyframeCloud_DS, indices);
        pcl::removeNaNFromPointCloud(*friendKeyframeCloud_DS, *friendKeyframeCloud_DS, indices);

        // icp
        pcl::IterativeClosestPoint<PointType, PointType> icp;
        // // gicp
        // pcl::GeneralizedIterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(2 * historyKeyframeSearchNum * curr_robot->keyframeMeterGap);
        icp.setMaximumIterations(50);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-1);

        icp.setInputSource(currKeyframeCloud_DS);
        icp.setInputTarget(friendKeyframeCloud_DS);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());

        Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();

        Pose6D loopKfPose = friend_robot->keyframePoses[friend_node_idx];
        // Eigen::Affine3f W_T_loopKF = friend_robot->W_T_odom * toEigenAffine3f(loopKfPose);
        // Eigen::Affine3f W_T_cur = W_T_odom * toEigenAffine3f(curr_robot->keyframePoses[curr_node_idx]);

        // Eigen::Matrix4f pose_diff = (W_T_loopKF.inverse() * W_T_cur).matrix();
        // std::cout << "pose_diff: "  << "\n" << pose_diff << std::endl;

        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
        if (useQuatro) {
            // std::cout << "Using Quatro for initial guess" << std::endl;
            std::lock_guard<std::mutex> lock(matchMutex);
            Eigen::Matrix4d quatro_output;

            pcl::PointCloud<PointType>::Ptr cur_nonground(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr friend_noground(new pcl::PointCloud<PointType>());

            matchPointClouds(currKeyframeCloud, friendKeyframeCloud, quatroParams, lidarType, neighborSelectionMode, 
                            groundSegMode, voxel_size, normal_radius, fpfh_radius, quatro_output, csf_params, 
                            cur_nonground, friend_noground);

            // matchPointClouds(currKeyframeCloud, friendKeyframeCloud, quatroParams, lidarType, neighborSelectionMode, 
                            // groundSegMode, voxel_size, normal_radius, fpfh_radius, quatro_output, csf_params);
            initial_guess = quatro_output.cast<float>();
        } 
        // else {
        //     // Use the original method to calculate the initial guess
        //     Eigen::Affine3f W_T_loopKF = friend_robot->W_T_odom * toEigenAffine3f(loopKfPose);
        //     Eigen::Affine3f W_T_cur = W_T_odom * toEigenAffine3f(curr_robot->keyframePoses[curr_node_idx]);
        //     initial_guess = (W_T_loopKF.inverse() * W_T_cur).matrix();
        // }

        std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        // std::cout << "initial_guess time: " << elapsed_seconds.count() << "s\n";

        icp.align(*unused_result, initial_guess);
        float rotang = diffRotation(initial_guess.block<3, 3>(0, 0), icp.getFinalTransformation().block<3, 3>(0, 0));

        if (!icp.hasConverged() || icp.getFitnessScore() > curr_robot->Multi_loopFitnessScoreThreshold) {
            // ROS_INFO_STREAM("[Inter SC loop] ICP fitness test failed (" << icp.getFitnessScore() << " > " << Multi_loopFitnessScoreThreshold << "). Reject this SC loop.");
            return std::nullopt;
        } else {
            ROS_INFO_STREAM("[ loop "<<cur_global_idx<<"-"<<friend_global_idx<<"] ICP fitness test passed (" 
                            << icp.getFitnessScore() << " < " << curr_robot->Multi_loopFitnessScoreThreshold
                                                                        << ", rot angle change " << rotang * 180 / M_PI << "). Add this SC loop.");
            {
                std::lock_guard<std::mutex> lock(*curr_robot->interloopPairMutex);
                curr_robot->interRobotLoopIndexContainer[curr_node_idx] = std::make_pair(friend_robot->robot_id, friend_node_idx);
            }

            sensor_msgs::PointCloud2 currKeyframeCloudRegMsg;
            pcl::PointCloud<PointType>::Ptr currKeyframeCloudWorldReg(new pcl::PointCloud<PointType>());
            currKeyframeCloudWorldReg = local2global(unused_result, loopKfPose);
            pcl::toROSMsg(*currKeyframeCloudWorldReg, currKeyframeCloudRegMsg);
            currKeyframeCloudRegMsg.header.frame_id = "camera_init";
            curr_robot->pub_multiLoopScanLocalRegisted.publish(currKeyframeCloudRegMsg);
        }

        logRotationAngleAndRobots(curr_robot->robot_id, friend_robot->robot_id, initial_guess, icp.getFitnessScore());

        Eigen::Affine3f correctionLidarFrame(icp.getFinalTransformation());
        Eigen::Matrix3f rot = correctionLidarFrame.rotation();
        Eigen::Vector3f trans = correctionLidarFrame.translation();

        // Eigen::Matrix3f rot = initial_guess.block<3, 3>(0, 0);
        // Eigen::Vector3f trans = initial_guess.block<3, 1>(0, 3);

        std::cout << "rot: "  << "\n" << rot << std::endl;
        std::cout << "trans: "  << "\n" << trans << std::endl;

        return gtsam::Pose3(gtsam::Rot3(rot.cast<double>()), trans.cast<double>());
    }

    // std::optional<gtsam::Pose3> doICPVirtualRelativeFriendRobot(Robot* curr_robot, Robot* friend_robot, int curr_node_idx, int friend_node_idx)
    // {
    //     // 确保关键帧索引有效
    //     if (curr_node_idx >= curr_robot->keyframePoses.size() || friend_node_idx >= friend_robot->keyframePoses.size()) {
    //         ROS_ERROR("Node index out of bounds");
    //         return std::nullopt;
    //     }

    //     // 初始化点云
    //     pcl::PointCloud<PointType>::Ptr currKeyframeCloud(new pcl::PointCloud<PointType>());
    //     pcl::PointCloud<PointType>::Ptr currKeyframeCloud_DS(new pcl::PointCloud<PointType>());
    //     pcl::PointCloud<PointType>::Ptr friendKeyframeCloud(new pcl::PointCloud<PointType>());
    //     pcl::PointCloud<PointType>::Ptr friendKeyframeCloud_DS(new pcl::PointCloud<PointType>());

    //     curr_robot->loopFindNearKeyframesCloud(currKeyframeCloud, currKeyframeCloud_DS, 
    //                                         curr_node_idx, historyKeyframeSearchNum, curr_node_idx);
    //     friend_robot->loopFindNearKeyframesCloud(friendKeyframeCloud, friendKeyframeCloud_DS, 
    //                                             friend_node_idx, historyKeyframeSearchNum, friend_node_idx);

    //     // 全局索引
    //     int cur_global_idx = curr_node_idx + curr_robot->index_offset;
    //     int friend_global_idx = friend_node_idx + friend_robot->index_offset;

    //     // 移除NaN点
    //     std::vector<int> indices;
    //     pcl::removeNaNFromPointCloud(*currKeyframeCloud_DS, *currKeyframeCloud_DS, indices);
    //     pcl::removeNaNFromPointCloud(*friendKeyframeCloud_DS, *friendKeyframeCloud_DS, indices);

    //     // ICP配置
    //     pcl::IterativeClosestPoint<PointType, PointType> icp;
    //     icp.setMaxCorrespondenceDistance(2 * historyKeyframeSearchNum * curr_robot->keyframeMeterGap);
    //     icp.setMaximumIterations(50);
    //     icp.setTransformationEpsilon(1e-6);
    //     icp.setEuclideanFitnessEpsilon(1e-1);
    //     icp.setInputSource(currKeyframeCloud_DS);
    //     icp.setInputTarget(friendKeyframeCloud_DS);
    //     pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());

    //     Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();
    //     Pose6D loopKfPose = friend_robot->keyframePoses[friend_node_idx];

    //     bool choose_quatro = false;

    //     // 关键帧计数逻辑
    //     if (useQuatro && (curr_robot->keyframePoses.size() % 20 == 0 && curr_robot->useQuatroCount < 5)) {
    //         // 使用Quatro进行初值计算
    //         std::lock_guard<std::mutex> lock(matchMutex);
    //         Eigen::Matrix4d quatro_output;
    //         matchPointClouds(currKeyframeCloud, friendKeyframeCloud, quatroParams, lidarType, neighborSelectionMode, 
    //                         groundSegMode, voxel_size, normal_radius, fpfh_radius, quatro_output, csf_params);
    //         initial_guess = quatro_output.cast<float>();
            
    //         choose_quatro == true; 
    //     } else {
    //         // 使用原始方法计算初值
    //         Eigen::Affine3f W_T_loopKF = friend_robot->W_T_odom * toEigenAffine3f(loopKfPose);
    //         Eigen::Affine3f W_T_cur = W_T_odom * toEigenAffine3f(curr_robot->keyframePoses[curr_node_idx]);
    //         initial_guess = (W_T_loopKF.inverse() * W_T_cur).matrix();
    //     }

    //     icp.align(*unused_result, initial_guess);
    //     float rotang = diffRotation(initial_guess.block<3, 3>(0, 0), icp.getFinalTransformation().block<3, 3>(0, 0));

    //     if (!icp.hasConverged() || icp.getFitnessScore() > curr_robot->Multi_loopFitnessScoreThreshold) {
    //         return std::nullopt;
    //     } else {
    //         ROS_INFO_STREAM("[ loop "<<cur_global_idx<<"-"<<friend_global_idx<<"] ICP fitness test passed (" 
    //                         << icp.getFitnessScore() << " < " << curr_robot->Multi_loopFitnessScoreThreshold
    //                         << ", rot angle change " << rotang * 180 / M_PI << "). Add this SC loop.");
    //         {
    //             std::lock_guard<std::mutex> lock(*curr_robot->interloopPairMutex);
    //             curr_robot->interRobotLoopIndexContainer[curr_node_idx] = std::make_pair(friend_robot->robot_id, friend_node_idx);
    //         }
    //     }


    //     if (choose_quatro) {
    //         useQuatroCount++;
    //     }
        
    //     Eigen::Affine3f correctionLidarFrame(icp.getFinalTransformation());
    //     Eigen::Matrix3f rot = correctionLidarFrame.rotation();
    //     Eigen::Vector3f trans = correctionLidarFrame.translation();

    //     return gtsam::Pose3(gtsam::Rot3(rot.cast<double>()), trans.cast<double>());
    // }

    void process_pg()
    {
        while (1)
        {
            // std::cout<<"process_pg still running"<<std::endl;
            while (!odometryBuf.empty() && !fullResBuf.empty())
            {
                // std::cout<< odometryBuf.size() <<" " <<fullResBuf.size()<<std::endl;
                std::lock_guard<std::mutex> lock(*mBuf);
                while (!odometryBuf.empty() && !fullResBuf.empty())
                {
                    if (odometryBuf.front()->header.stamp + ros::Duration(1e-4) < fullResBuf.front()->header.stamp)
                        odometryBuf.pop();
                    else if (odometryBuf.front()->header.stamp > fullResBuf.front()->header.stamp + ros::Duration(1e-4))
                        fullResBuf.pop();
                    else
                        break;
                }
                if (odometryBuf.empty() || fullResBuf.empty())
                {
                    break;
                }

                timeLaserOdometry = odometryBuf.front()->header.stamp;
                timeLaser = fullResBuf.front()->header.stamp;

                if (timeLaser != timeLaserOdometry)
                {
                    double d = (timeLaser - timeLaserOdometry).toSec();
                    ROS_INFO_STREAM("Large time gap between deskewed laser scan and odometry pose " << d << " sec.");
                }

                laserCloudFullRes->clear();
                pcl::PointCloud<PointType>::Ptr thisKeyFrame(new pcl::PointCloud<PointType>());
                pcl::fromROSMsg(*fullResBuf.front(), *thisKeyFrame);
                fullResBuf.pop();
                Pose6D pose_curr = getOdom(odometryBuf.front());
                odometryBuf.pop();

                double eps = 0.1;
                while (!gpsBuf.empty())
                {
                    auto thisGPS = gpsBuf.front();
                    auto thisGPSTime = thisGPS->header.stamp;
                    if (fabs((thisGPSTime - timeLaserOdometry).toSec()) < eps)
                    {
                        currGPS = thisGPS;
                        hasGPSforThisKF = true;
                        break;
                    }
                    else
                    {
                        hasGPSforThisKF = false;
                    }
                    gpsBuf.pop();
                }

                // TODO yiwem: add dense odom in DenseframePoses
                // mtxPosegraph_dense->lock();
                // DenseframePoses.push_back(pose_curr);
                // DenseframeTimes.push_back(timeLaserOdometry);
                // mtxPosegraph_dense->unlock();

                allframeLaserClouds.push_back(thisKeyFrame);
                allframePoses.push_back(pose_curr);

                odom_pose_prev = odom_pose_curr;
                odom_pose_curr = pose_curr;
                Pose6D dtf = diffTransformation(odom_pose_prev, odom_pose_curr);

                double delta_translation = sqrt(dtf.x * dtf.x + dtf.y * dtf.y + dtf.z * dtf.z);
                translationAccumulated += delta_translation;
                rotationAccumulated += (dtf.roll + dtf.pitch + dtf.yaw);

                if (translationAccumulated > keyframeMeterGap || rotationAccumulated > keyframeRadGap || addAllframes)
                {
                    isNowKeyFrame = true;
                    translationAccumulated = 0.0;
                    rotationAccumulated = 0.0;
                }
                else
                {
                    isNowKeyFrame = false;
                }

                if (!isNowKeyFrame)
                    continue;

                if (!gpsOffsetInitialized)
                {
                    if (hasGPSforThisKF)
                    {
                        gpsAltitudeInitOffset = currGPS->altitude;
                        gpsOffsetInitialized = true;
                    }
                }

                mKF->lock();


                keyframePoses.push_back(pose_curr);
                keyframePosesUpdated.push_back(pose_curr);
                keyframeTimes.push_back(timeLaserOdometry);

                // std::cout << "allframeLaserClouds.size(): " << allframeLaserClouds.size() << std::endl;
                keyframeIndexTOallID.insert(std::pair<int, int>(keyframePoses.size()-1, allframeLaserClouds.size()-1));

                pcl::PointCloud<PointType>::Ptr thisKeyFrameDS(new pcl::PointCloud<PointType>());
                downSizeFilterLoopClosure.setInputCloud(thisKeyFrame);
                downSizeFilterLoopClosure.filter(*thisKeyFrameDS);

                keyframeLaserClouds.push_back(thisKeyFrameDS);
                
                if(lcd_mode==1){
                    // Add last scan in scancontext database
                    scMutex->lock();
                    scManager.makeAndSaveScancontextAndKeys(*thisKeyFrameDS);
                    scMutex->unlock();
                }
                else if(lcd_mode==2){
                    int cur_submap_index = keyframePoses.size()-1;
                    pcl::PointCloud<PointType>::Ptr thisKeySubmap(new pcl::PointCloud<PointType>());

                    // TODO binliang: 无需用keyframePoses.size()和halfsubmapsize对比吧？
                    if(keyframePoses.size()-1>=halfsubmapsize){ // update descriptors for submapsize
                        cur_submap_index = keyframePoses.size()-1-halfsubmapsize; 
                        TicTocV2 t_make_submap(false);
                        thisKeySubmap->clear();
                        // std::cout << "cur_submap_index: " << cur_submap_index << std::endl;
                        // std::cout << "keyframeIndexTOallID[cur_submap_index]: " << keyframeIndexTOallID[cur_submap_index] << std::endl;
                        // loopFindNearKeyframesCloud2(thisKeySubmap, cur_submap_index, halfsubmapsize, cur_submap_index); 
                        loopFindNearframesCloud(thisKeySubmap, keyframeIndexTOallID[cur_submap_index], halfsubmapsize, keyframeIndexTOallID[cur_submap_index]);
                        
                        t_make_submap.toc("make submap");
                                        
                        DeepDescMutex->lock();
                        deepDescManager.makeAndSaveDeepDescAndKeys(*thisKeySubmap);
                        DeepDescMutex->unlock();
                        // deepDescManager.remakeAndSaveDeepDescAndKeysAtIndex(*thisKeySubmap, cur_submap_index);
                        pcl::io::savePCDFileBinary(saveSubmapDirectory + padZeros(cur_submap_index) + ".pcd", *thisKeySubmap); // scan 
                    } 
                }

                *hasNewScanForLC = true;
                mKF->unlock();

                const int init_node_idx = 0;
                gtsam::Pose3 poseOrigin = Pose6DtoGTSAMPose3(keyframePoses.at(init_node_idx));

                const int prev_node_idx = keyframePoses.size() - 2;
                const int curr_node_idx = keyframePoses.size() - 1;
                gtsam::Pose3 poseFrom;
                if (prev_node_idx >= 0)
                    poseFrom = Pose6DtoGTSAMPose3(keyframePoses.at(prev_node_idx));
                gtsam::Pose3 poseTo = Pose6DtoGTSAMPose3(keyframePoses.at(curr_node_idx));

                if (!(*gtSAMgraphMade))
                {
                    if(robot_id==0){
                        mtxPosegraph->lock();
                        {
                            gtSAMgraph->add(gtsam::PriorFactor<gtsam::Pose3>(index_offset + init_node_idx, poseOrigin, priorNoise));
                            initialEstimate.insert(index_offset + init_node_idx, poseOrigin);
                            writeVertex(init_node_idx, poseOrigin);
                            newStateTimes.push_back(timeLaser);
                        }
                        mtxPosegraph->unlock();
                        cout << "posegraph prior node " << init_node_idx << " added, robot 0." <<keyframePoses.size()<< endl;
                    }
                    else{
                        mtxPosegraph->lock();
                        {
                            const int curr_global_node_idx = index_offset + curr_node_idx;

                            // // for WHU data
                            // gtSAMgraph->add(gtsam::PriorFactor<gtsam::Pose3>(curr_global_node_idx, poseOrigin, priorNoise));
                            // poseTo = Pose6DtoGTSAMPose3(Pose6D{3, 3, 0, 0, 0, 0});
                            // auto relative_pose = poseOrigin.between(poseTo);

                            // for rooftop data
                            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
                            // if (robot_id == 1){
                            //     T << 0.725996553898, -0.687698364258, 0.000000000000, 12.159266471863,
                            //     0.687698364258, 0.725996553898, 0.000000000000, -7.136914253235,
                            //     0.000000000000, 0.000000000000, 1.000000000000, 0.030006254092,
                            //     0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000;}
                            // else if (robot_id == 2){
                            //     T << 0.648171961308, 0.761493980885, 0.000000000000, 20.416875839233,
                            //     -0.761493980885, 0.648171961308, 0.000000000000, -26.168693542480,
                            //     0.000000000000, 0.000000000000, 1.000000000000, 0.000000000000,
                            //     0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000;}
                            W_T_odom = Eigen::Affine3d(T).cast<float>();
                            poseTo = gtsam::Pose3(gtsam::Rot3(T.block<3, 3>(0, 0)), gtsam::Point3(T(0, 3), T(1, 3), T(2, 3)));
                            auto relative_pose = poseTo;

                            gtSAMgraph->add(gtsam::BetweenFactor<gtsam::Pose3>(0, curr_global_node_idx, relative_pose, initTw_Noise));
                            initialEstimate.insert(curr_global_node_idx, poseOrigin);
                            writeVertex(curr_node_idx, poseTo);
                            newStateTimes.push_back(timeLaser);
                            // cout << "first posegraph node " << curr_global_node_idx << " added, robot-" << robot_id <<". "<<keyframePoses.size()<< endl;
                        }
                        mtxPosegraph->unlock();
                    }
                    *gtSAMgraphMade = true;
                }
                else
                {
                    mtxRecentPose->lock();
                    double copyRecentOptimizedX = recentOptimizedX;
                    double copyRecentOptimizedY = recentOptimizedY;
                    mtxRecentPose->unlock();
                    mtxPosegraph_isam->lock();
                    mtxPosegraph->lock();
                    {
                        const int curr_global_node_idx = index_offset + curr_node_idx;
                        const int prev_global_node_idx = index_offset + prev_node_idx;
                        // std::cout<<"curr_global_node_idx: "<<curr_global_node_idx<< " prev_global_node_idx: "<<prev_global_node_idx<<" keyframePoses.size(): "<<keyframePoses.size()<<std::endl;
                        gtSAMgraph->add(gtsam::BetweenFactor<gtsam::Pose3>(prev_global_node_idx, curr_global_node_idx, poseFrom.between(poseTo), odomNoise));
                        writeEdge({prev_node_idx, curr_node_idx}, poseFrom.between(poseTo));
                        if (hasGPSforThisKF)
                        {
                            double curr_altitude_offseted = currGPS->altitude - gpsAltitudeInitOffset;
                            gtsam::Point3 gpsConstraint(copyRecentOptimizedX, copyRecentOptimizedY, curr_altitude_offseted);
                            gtSAMgraph->add(gtsam::GPSFactor(curr_global_node_idx, gpsConstraint, robustGPSNoise));
                            cout << "GPS factor added at node " << curr_node_idx << endl;
                        }

                        initialEstimate.insert(curr_global_node_idx, poseTo);
                        writeVertex(curr_node_idx, poseTo);
                        newStateTimes.push_back(timeLaser);
                    }
                    mtxPosegraph->unlock();
                    mtxPosegraph_isam->unlock();

                    if (curr_node_idx % 100 == 0)
                        cout << "posegraph odom node " << curr_node_idx << " added." << endl;
                }
                std::mutex file_mutex;
                std::lock_guard<std::mutex> file_lock(file_mutex);
                std::string save_pcd_index = "scan_" + std::to_string(curr_node_idx);
                pcl::io::savePCDFileBinary(pgScansDirectory + save_pcd_index + ".pcd", *thisKeyFrame);
                pgTimeSaveStream << timeLaser << std::endl;

                if (lcd_mode == 1)
                {
                    const auto &curr_scd = scManager.getConstRefRecentSCD();
                    if (curr_node_idx != scManager.polarcontexts_.size() - 1)
                    {
                        std::cerr << "Inconsistent scan context sizes curr_node_idx + 1: " << curr_node_idx + 1
                                << ", scManager.polarcontexts_.size(): " << scManager.polarcontexts_.size() << std::endl;
                    }
                    saveSCD(saveSCDDirectory + save_pcd_index + ".scd", curr_scd);
                }
            }

            std::chrono::milliseconds dura(2);
            std::this_thread::sleep_for(dura);
            // *shutdown = terminator.quit();
            // *shutdown = false;
            if (*shutdown || true)
            {
                // cout << "****************************************************" << endl;
                // cout << "Saving the posegraph for" << robot_id << " ..." << endl;
                for (auto &_line : vertices_str)
                    pgSaveStream << _line << std::endl;
                for (auto &_line : edges_str)
                    pgSaveStream << _line << std::endl;
                pgSaveStream.close();

                pgTimeSaveStream.close();
                scanMatchStream.close();
                robots_scanMatchStream.close();
                // break;
            }
        }
    }

    void performSCLoopClosure()
    {
        mKF->lock();
        int numPoses = int(keyframePoses.size());
        mKF->unlock();
        if (numPoses < scManager.NUM_EXCLUDE_RECENT)
            return;
        scMutex->lock();
        auto detectResult = scManager.detectLoopClosureID();
        scMutex->unlock();
        int SCclosestHistoryFrameID = detectResult.first;
        // std::cout << "SCclosestHistoryFrameID: " << SCclosestHistoryFrameID << std::endl;
        if (SCclosestHistoryFrameID != -1)
        {
            const int prev_node_idx = SCclosestHistoryFrameID;
            const int curr_node_idx = numPoses - 1;

            mBuf->lock();
            initLoopICPBuf.push(std::pair<int, int>(prev_node_idx, curr_node_idx));
            mBuf->unlock();
        }
    }

    void perfromDeepDescLoopClosure(void){
        mKF->lock();
        int numPoses = int(keyframePoses.size());
        mKF->unlock();

        if( int(numPoses-halfsubmapsize) < deepDescManager.NUM_EXCLUDE_RECENT) // do not try too early 
            return;

        DeepDescMutex->lock();
        auto detectResult = deepDescManager.detectLoopClosureID(); // first: nn index, second: yaw diff 
        DeepDescMutex->unlock();

        int closestHistoryFrameID = detectResult.first;

        if( closestHistoryFrameID != -1 ) { 
            const int prev_node_idx = closestHistoryFrameID;

            // std::cout << "numposes: " << numPoses << std::endl;
            // std::cout << "deepdesc_invkeys_mat_.size:" << deepDescManager.deepdesc_invkeys_mat_.size() << std::endl;
            
            // const int curr_node_idx = keyframePoses.size() - 1 - halfsubmapsize; // because cpp starts 0 and ends n-1
            const int curr_node_idx = deepDescManager.deepdesc_invkeys_mat_.size() - 1; // because cpp starts 0 and ends n-1

            // std::cout << "Loop between " << prev_node_idx << " and " << curr_node_idx << " for robot " << robot_id << ", time diff: " 
            //         <<  fabs((keyframeTimes[prev_node_idx] - keyframeTimes[curr_node_idx]).toSec())<< std::endl;
            
            // time diff judge
            // if (fabs((keyframeTimes[prev_node_idx] - keyframeTimes[curr_node_idx]).toSec()) < historyKeyframeSearchTimeDiff)
            // {
                    //     return;
            // }

            // space diff judge
            if (sqrt((keyframePoses[prev_node_idx].x - keyframePoses[curr_node_idx].x)*(keyframePoses[prev_node_idx].x - keyframePoses[curr_node_idx].x) + 
                    (keyframePoses[prev_node_idx].y - keyframePoses[curr_node_idx].y)*(keyframePoses[prev_node_idx].y - keyframePoses[curr_node_idx].y) + 
                    (keyframePoses[prev_node_idx].z - keyframePoses[curr_node_idx].z)*(keyframePoses[prev_node_idx].z - keyframePoses[curr_node_idx].z)) > historyKeyframeSearchRadius)
            {
                cout << "Loop invalid becaues space diff large." << endl;
                return;
            }

            mBuf->lock();
            initLoopICPBuf.push(std::pair<int, int>(prev_node_idx, curr_node_idx));
            mBuf->unlock();
        }
    }
    

    bool detectLoopClosureDistance(int *loopKeyCur, int *loopKeyPre)
    {
        mKF->lock();
        pcl::PointCloud<pcl::PointXYZ>::Ptr copy_cloudKeyPoses3D = vector2pc(keyframePoses);
        mKF->unlock();
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;

        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);

        mKF->lock();
        for (size_t i = 0; i < pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            if (std::fabs((keyframeTimes[id] - keyframeTimes[*loopKeyCur]).toSec()) > historyKeyframeSearchTimeDiff)
            {
                *loopKeyPre = id;
                break;
            }
        }
        mKF->unlock();

        if (*loopKeyPre == -1 || *loopKeyCur == *loopKeyPre)
            return false;

        return true;
    }


    void performRSLoopClosure()
    {
        if (!(*hasNewScanForLC))
            return;
        else
            *hasNewScanForLC = false;
        mKF->lock();
        int numPoses = int(keyframePoses.size());
        mKF->unlock();
        if (numPoses < scManager.NUM_EXCLUDE_RECENT)
            return;

        int loopKeyCur = numPoses - 1;
        int loopKeyPre = -1;
        if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre))
        {
            // cout << "Loop candidate between " << loopKeyPre << " and " << loopKeyCur << "" << endl;
            mBuf->lock();
            initLoopICPBuf.push(std::pair<int, int>(loopKeyPre, loopKeyCur));
            mBuf->unlock();
        }
    }


    void performFriendSCLoopClosure() {
        mKF->lock();
        int numPoses = int(keyframePoses.size());
        mKF->unlock();
        
        if (numPoses < scManager.NUM_EXCLUDE_RECENT)
            return;

        for (auto& [friend_robot_id, friend_robot] : friend_robots) {
            if(std::try_lock(*scMutex, *friend_robot->scMutex) == -1){
                auto detectResult = detectSCLoopIDFriendRobot(*this, friend_robot);
                friend_robot->scMutex->unlock();
                scMutex->unlock();

                int curr_robot_id = std::get<0>(detectResult);
                // int friend_robot_id = std::get<1>(detectResult);
                int loop_id = std::get<2>(detectResult);
                float yaw_diff_rad = std::get<3>(detectResult);


                if (loop_id != -1) {
                    const int curr_node_idx = numPoses - 1;
                    const int friend_node_idx = loop_id;

                    // Save SC initial guess
                    // {
                    //     std::lock_guard<std::mutex> lock(scGuessMutex);
                    //     Eigen::Affine3f scInitialGuess = Eigen::Affine3f::Identity();
                    //     scInitialGuess.rotate(Eigen::AngleAxisf(yaw_diff_rad, Eigen::Vector3f::UnitZ()));
                    //     scInitialGuess.translation() << 0, 0, 0;  // Assuming zero translation           
                    //     this->scInitialGuess = scInitialGuess;
                    // }
                    mBuf->lock();
                    interRobotLoopICPBuf.push(std::make_tuple(curr_robot_id, friend_robot_id, curr_node_idx, friend_node_idx));
                    mBuf->unlock();
                }
            }
        }
    }

    std::tuple<int, int, int, float> detectSCLoopIDFriendRobot(Robot& robot, std::shared_ptr<Robot>& friend_robot) {
        int loop_id = -1;
        int friend_robot_idx = -1;
        float yaw_diff_rad = 0.0;

        if (friend_robot->scManager.polarcontext_invkeys_mat_.empty() || friend_robot->scManager.polarcontext_invkeys_mat_[0].empty()) {
            ROS_WARN("polarcontext_invkeys_mat_ is empty for friend_robot_id: %d", friend_robot->robot_id);
            return std::make_tuple(robot.robot_id, friend_robot->robot_id, loop_id, yaw_diff_rad);
        }

        double min_dist = std::numeric_limits<double>::max();

        
        // 重新构建树的条件
        if (!friend_robot->scManager.is_tree_batch_made 
            || (friend_robot->tree_making_counter % TREE_MAKING_PERIOD == 0)
            ) {
            friend_robot->scManager.polarcontext_invkeys_to_search_.clear();
            friend_robot->scManager.polarcontext_invkeys_to_search_.assign(
                friend_robot->scManager.polarcontext_invkeys_mat_.begin(),
                friend_robot->scManager.polarcontext_invkeys_mat_.end()
            );

            friend_robot->scManager.polarcontext_tree_batch_ = std::make_unique<InvKeyTree>(
                friend_robot->scManager.PC_NUM_RING /* dim */,
                friend_robot->scManager.polarcontext_invkeys_to_search_,
                10 /* max leaf */
            );
            friend_robot->scManager.is_tree_batch_made = true;
        }
        friend_robot->tree_making_counter += 1;


        // return std::make_tuple(robot.robot_id, friend_robot->robot_id, loop_id, yaw_diff_rad);

        if (robot.scManager.polarcontext_invkeys_mat_.empty() || robot.scManager.polarcontext_invkeys_mat_.back().empty()) {
            ROS_WARN("robot.scManager.polarcontext_invkeys_mat_ is empty for robot_id: %d", robot.robot_id);
            return std::make_tuple(robot.robot_id, friend_robot->robot_id, loop_id, yaw_diff_rad);
        }

        auto curr_key = robot.scManager.polarcontext_invkeys_mat_.back();
        auto curr_desc = robot.scManager.polarcontexts_.back();

        if (curr_key.empty() || curr_desc.rows() == 0) {
            ROS_WARN("Current key or descriptor is empty for robot");
            return std::make_tuple(robot.robot_id, friend_robot->robot_id, loop_id, yaw_diff_rad);
        }

        std::vector<size_t> candidate_indexes(friend_robot->scManager.NUM_CANDIDATES_FROM_TREE);
        std::vector<float> out_dists_sqr(friend_robot->scManager.NUM_CANDIDATES_FROM_TREE);

        nanoflann::KNNResultSet<float> knnsearch_result(friend_robot->scManager.NUM_CANDIDATES_FROM_TREE);
        knnsearch_result.init(&candidate_indexes[0], &out_dists_sqr[0]);

        friend_robot->scManager.polarcontext_tree_batch_->index->findNeighbors(knnsearch_result, &curr_key[0], nanoflann::SearchParams(10));

        for (int candidate_iter_idx = 0; candidate_iter_idx < scManager.NUM_CANDIDATES_FROM_TREE; candidate_iter_idx++) {
            if(candidate_indexes[candidate_iter_idx]<0 || candidate_indexes[candidate_iter_idx]>=friend_robot->scManager.polarcontexts_.size()){
                continue;
            }
            MatrixXd polarcontext_candidate = friend_robot->scManager.polarcontexts_[candidate_indexes[candidate_iter_idx]];
            std::pair<double, int> sc_dist_result = robot.scManager.distanceBtnScanContext(curr_desc, polarcontext_candidate);

            double candidate_dist = sc_dist_result.first;
            int candidate_align = sc_dist_result.second;

            if (candidate_dist < min_dist) {
                min_dist = candidate_dist;
                // yaw_diff_rad = deg2rad(candidate_align * scManager.PC_UNIT_SECTORANGLE);
                yaw_diff_rad = candidate_align * scManager.PC_UNIT_SECTORANGLE * M_PI / 180.0;
                loop_id = candidate_indexes[candidate_iter_idx];
                friend_robot_idx = friend_robot->robot_id;
            }
        }
        // if (robot.robot_id == 2 || friend_robot_idx == 2) {
        //             std::cout << "robot_id: " << robot.robot_id << " friend_robot_id: " << friend_robot->robot_id << " loop_id: " << loop_id << " min_dist: " << min_dist << std::endl;
        //         }

        if (min_dist < Multi_scDistThres) {
            // std::cout << "robot_id: " << robot.robot_id << " friend_robot_id: " << friend_robot->robot_id << " loop_id: " << loop_id << " min_dist: " << min_dist << std::endl;
            return std::make_tuple(robot.robot_id, friend_robot_idx, loop_id, yaw_diff_rad);
        }

        return std::make_tuple(robot.robot_id, friend_robot_idx, loop_id, yaw_diff_rad);
    }

    void performFriendDeepDescLoopClosure() {
        mKF->lock();
        int numPoses = int(keyframePoses.size());
        mKF->unlock();

        if (int(numPoses - halfsubmapsize) < deepDescManager.NUM_EXCLUDE_RECENT) {
            return;
        }

        for (auto& [friend_robot_id, friend_robot] : friend_robots) {
            if (std::try_lock(*DeepDescMutex, *friend_robot->DeepDescMutex) == -1) {
                auto detectResult = detectDeepLoopFriendRobot(*this, friend_robot);
                friend_robot->DeepDescMutex->unlock();
                DeepDescMutex->unlock();

                int curr_robot_id = std::get<0>(detectResult);
                int loop_id = std::get<2>(detectResult);
                float yaw_diff_rad = std::get<3>(detectResult);


                if (loop_id != -1) {
                    const int curr_node_idx = this->deepDescManager.deepdesc_invkeys_mat_.size() - 1;  // 当前节点索引
                    const int friend_node_idx = loop_id;

                    // if (sqrt(pow(friend_robot->keyframePoses[friend_node_idx].x - keyframePoses[curr_node_idx].x, 2) +
                    //         pow(friend_robot->keyframePoses[friend_node_idx].y - keyframePoses[curr_node_idx].y, 2) +
                    //         pow(friend_robot->keyframePoses[friend_node_idx].z - keyframePoses[curr_node_idx].z, 2)) > historyKeyframeSearchRadius) {
                    //     std::cout << "Loop invalid because space diff large." << std::endl;
                    //     return;
                    // }

                    std::cout << "Candidate valid! Loop detected between " << "Robot " << curr_robot_id << ": " << curr_node_idx << 
                            " and " << " Friend Robot " << friend_robot_id << ": " << friend_node_idx << std::endl;

                    mBuf->lock();
                    interRobotLoopICPBuf.push(std::make_tuple(curr_robot_id, friend_robot_id, curr_node_idx, friend_node_idx));
                    mBuf->unlock();
                }
            }
        }
    }


    std::tuple<int, int, int, float> detectDeepLoopFriendRobot(Robot& robot, std::shared_ptr<Robot>& friend_robot) {
        int loop_id = -1;
        int friend_robot_idx = -1;
        float yaw_diff_rad = 0.0;

        // std::cout << " friend_robot_id: " << friend_robot->robot_id << "tree_making_counter: " << friend_robot->tree_making_counter << std::endl;

        // 检查友机的描述子矩阵是否为空
        if (friend_robot->deepDescManager.deepdesc_invkeys_mat_.empty() || friend_robot->deepDescManager.deepdesc_invkeys_mat_[0].empty()) {
            ROS_WARN("deepdesc_invkeys_mat_ is empty for friend_robot_id: %d", friend_robot->robot_id);
            return std::make_tuple(robot.robot_id, friend_robot->robot_id, loop_id, yaw_diff_rad);
        }

        double min_dist = std::numeric_limits<double>::max();

        // 判断是否需要重新构建 KD-tree
        if (friend_robot->tree_making_counter % TREE_MAKING_PERIOD == 0) {
            friend_robot->deepDescManager.deepdesc_invkeys_to_search_.clear();

            for (auto iter = friend_robot->deepDescManager.deepdesc_invkeys_mat_.begin(); 
                    iter < friend_robot->deepDescManager.deepdesc_invkeys_mat_.end(); 
                    iter += friend_robot->deepDescManager.TREE_MAKE_STEP_) { 
                friend_robot->deepDescManager.deepdesc_invkeys_to_search_.push_back(*iter);
            }

            friend_robot->deepDescManager.deepdesc_tree_ = std::make_unique<InvKeyTree>(
                friend_robot->deepDescManager.outputDim_, 
                friend_robot->deepDescManager.deepdesc_invkeys_to_search_, 
                10 /* max leaf */
            );
            // friend_robot->deepDescManager.is_tree_made = true;
        }
        friend_robot->tree_making_counter += 1;

        // 获取当前机器人的最新描述子
        if (robot.deepDescManager.deepdesc_invkeys_mat_.empty() || robot.deepDescManager.deepdesc_invkeys_mat_.back().empty()) {
            ROS_WARN("robot.deepdesc_invkeys_mat_ is empty for robot_id: %d", robot.robot_id);
            return std::make_tuple(robot.robot_id, friend_robot->robot_id, loop_id, yaw_diff_rad);
        }

        if(last_inter_lcd_size==robot.deepDescManager.deepdesc_invkeys_mat_.size()){
            return std::make_tuple(robot.robot_id, friend_robot_idx, loop_id, yaw_diff_rad);
        }
        auto curr_key = robot.deepDescManager.deepdesc_invkeys_mat_.back();
        last_inter_lcd_size = robot.deepDescManager.deepdesc_invkeys_mat_.size();

        // KD-tree 最近邻搜索
        std::vector<size_t> candidate_indexes(friend_robot->deepDescManager.NUM_CANDIDATES_FROM_TREE);
        std::vector<float> out_dists_sqr(friend_robot->deepDescManager.NUM_CANDIDATES_FROM_TREE);

        nanoflann::KNNResultSet<float> knnsearch_result(friend_robot->deepDescManager.NUM_CANDIDATES_FROM_TREE);
        knnsearch_result.init(&candidate_indexes[0], &out_dists_sqr[0]);

        friend_robot->deepDescManager.deepdesc_tree_->index->findNeighbors(knnsearch_result, &curr_key[0], nanoflann::SearchParams(10));

        for (int candidate_iter_idx = 0; candidate_iter_idx < knnsearch_result.size(); candidate_iter_idx++) {
            int candidate_idx = deepDescManager.TREE_MAKE_STEP_ * candidate_indexes[candidate_iter_idx];
            float candidate_dist = out_dists_sqr[candidate_iter_idx];

            if (candidate_idx < 0 || candidate_idx >= friend_robot->deepDescManager.deepdesc_invkeys_mat_.size()) {
                continue;
            }

            if (candidate_dist < min_dist) {
                min_dist = candidate_dist;
                loop_id = candidate_idx;
                friend_robot_idx = friend_robot->robot_id;
            }
        }

        // 判断距离是否小于设定阈值
        if (min_dist < robot.deepDescManager.Multi_DeepLOOP_DIST_THRES) {
            // std::cout << "robot_id: " << robot.robot_id << " friend_robot_id: " << friend_robot->robot_id << " loop_id: " << loop_id << " min_dist: " << min_dist << std::endl;
            return std::make_tuple(robot.robot_id, friend_robot_idx, loop_id, yaw_diff_rad);
        }

        return std::make_tuple(robot.robot_id, friend_robot_idx, loop_id, yaw_diff_rad);
    }


    void visualizeLoopClosure()
    {
        loopPairMutex->lock();
        std::unordered_map<int, int> copyLoopIndexContainer = loopIndexContainer;
        loopPairMutex->unlock();
        if (copyLoopIndexContainer.empty())
            return;

        visualization_msgs::MarkerArray markerArray;
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = "camera_init";
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.5;
        markerNode.scale.y = 0.5;
        markerNode.scale.z = 0.5;
        markerNode.color.r = 0;
        markerNode.color.g = 0.8;
        markerNode.color.b = 1;
        markerNode.color.a = 1;

        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = "camera_init";
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.6;
        markerEdge.color.r = 0.9;
        markerEdge.color.g = 0.9;
        markerEdge.color.b = 0;
        markerEdge.color.a = 1;

        mKF->lock();
        markerNode.header.stamp = keyframeTimes.back();
        markerEdge.header.stamp = keyframeTimes.back();
        for (auto it = copyLoopIndexContainer.begin(); it != copyLoopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;

            geometry_msgs::Point p;
            // p.x = keyframePosesUpdated[key_cur].x;
            // p.y = keyframePosesUpdated[key_cur].y;
            // p.z = keyframePosesUpdated[key_cur].z;   // TODO yiwen: use W_T_odom * keyframePoses instead of keyframePosesUpdated
            Eigen::Affine3f curr_keyframePoses = W_T_odom * toEigenAffine3f(keyframePoses[key_cur]);
            p.x = curr_keyframePoses.translation().x();
            p.y = curr_keyframePoses.translation().y();
            p.z = curr_keyframePoses.translation().z();

            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);

            // p.x = keyframePosesUpdated[key_pre].x;
            // p.y = keyframePosesUpdated[key_pre].y;
            // p.z = keyframePosesUpdated[key_pre].z;
            Eigen::Affine3f prev_keyframePoses = W_T_odom * toEigenAffine3f(keyframePoses[key_pre]);
            p.x = prev_keyframePoses.translation().x();
            p.y = prev_keyframePoses.translation().y();
            p.z = prev_keyframePoses.translation().z();

            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }
        mKF->unlock();

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdge.publish(markerArray);
    }

    void visualizeInterRobotLoopClosure()
    {
        interloopPairMutex->lock();
        std::unordered_map<int, std::pair<int, int>> copyInterRobotLoopIndexContainer = interRobotLoopIndexContainer;
        interloopPairMutex->unlock();

        if (copyInterRobotLoopIndexContainer.empty())
            return;

        visualization_msgs::MarkerArray markerArray;
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = "camera_init";
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "inter_loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.5;
        markerNode.scale.y = 0.5;
        markerNode.scale.z = 0.5;
        markerNode.color.r = 1.0; // red for inter-robot loops
        markerNode.color.g = 0.0;
        markerNode.color.b = 0.0;
        markerNode.color.a = 1;

        visualization_msgs::Marker interMarkerEdge;
        interMarkerEdge.header.frame_id = "camera_init";
        interMarkerEdge.action = visualization_msgs::Marker::ADD;
        interMarkerEdge.type = visualization_msgs::Marker::LINE_LIST;
        interMarkerEdge.ns = "inter_loop_edges";
        interMarkerEdge.id = 1;
        interMarkerEdge.pose.orientation.w = 1;
        interMarkerEdge.scale.x = 0.6;
        interMarkerEdge.color.r = 1.0;
        interMarkerEdge.color.g = 0.0;
        interMarkerEdge.color.b = 0.0;
        interMarkerEdge.color.a = 1;

        mKF->lock();
        markerNode.header.stamp = keyframeTimes.back();
        interMarkerEdge.header.stamp = keyframeTimes.back();

        // 多机闭环可视化
        for (auto it = copyInterRobotLoopIndexContainer.begin(); it != copyInterRobotLoopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;

            int friend_robot_id = it->second.first;
            int friend_node_idx = it->second.second;
            geometry_msgs::Point p;
            // p.x = keyframePosesUpdated[key_cur].x;
            // p.y = keyframePosesUpdated[key_cur].y;
            // p.z = keyframePosesUpdated[key_cur].z;  // TODO yiwen: use W_T_odom * keyframePoses instead of keyframePosesUpdated

            Eigen::Affine3f curr_keyframePoses = W_T_odom * toEigenAffine3f(keyframePoses[key_cur]);
            p.x = curr_keyframePoses.translation().x();
            p.y = curr_keyframePoses.translation().y();
            p.z = curr_keyframePoses.translation().z();

            markerNode.points.push_back(p);
            interMarkerEdge.points.push_back(p);
            // p.x = friend_robots[friend_robot_id]->keyframePosesUpdated[friend_node_idx].x;
            // p.y = friend_robots[friend_robot_id]->keyframePosesUpdated[friend_node_idx].y;
            // p.z = friend_robots[friend_robot_id]->keyframePosesUpdated[friend_node_idx].z;
            Eigen::Affine3f friend_keyframePoses = friend_robots[friend_robot_id]->W_T_odom * toEigenAffine3f(friend_robots[friend_robot_id]->keyframePoses[friend_node_idx]);
            p.x = friend_keyframePoses.translation().x();
            p.y = friend_keyframePoses.translation().y();
            p.z = friend_keyframePoses.translation().z();

            markerNode.points.push_back(p);
            interMarkerEdge.points.push_back(p);
        }

        mKF->unlock();

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(interMarkerEdge);
        pub_multiLoopConstraintEdge.publish(markerArray);
    }


    void process_lcd()
    {
        // float loopClosureFrequency = 1.0;
        ros::Rate rate(loopClosureFrequency);
        while (ros::ok())
        {
            rate.sleep();
            // std::cout << "process_lcd still running" << std::endl;
            if (lcd_mode == 0){
                std::cout << "radius search" << std::endl;
                performRSLoopClosure();
            }
            else if (lcd_mode == 1){
                performSCLoopClosure();
                performFriendSCLoopClosure();
            }
            else if (lcd_mode == 2){
                perfromDeepDescLoopClosure();
                performFriendDeepDescLoopClosure();
            }

            visualizeLoopClosure();
            visualizeInterRobotLoopClosure();
            
            // if (*shutdown)
            //     break;
        }
    }

    void process_icp()
    {
        while (ros::ok())
        {
            while (!initLoopICPBuf.empty())
            {
                // std::lock_guard<std::mutex> lock(*mBuf);
                if (initLoopICPBuf.size() > 30)
                {
                    ROS_WARN("Too many loop clousre candidates to be ICPed is waiting ... Do process_lcd less frequently (adjust loopClosureFrequency)");
                }

                std::pair<int, int> loop_idx_pair = initLoopICPBuf.front();
                mBuf->lock();
                initLoopICPBuf.pop();
                mBuf->unlock();

                const int prev_node_idx = loop_idx_pair.first;
                const int curr_node_idx = loop_idx_pair.second;
                auto relative_pose_optional = doICPVirtualRelative(prev_node_idx, curr_node_idx);
                if (relative_pose_optional)
                {
                    gtsam::Pose3 relative_pose = relative_pose_optional.value();
                    // std::lock_guard<std::mutex> lock(*mtxPosegraph);
                    mtxPosegraph->lock();
                    const int curr_global_node_idx = index_offset + curr_node_idx;
                    const int prev_global_node_idx = index_offset + prev_node_idx;
             
                    // std::cout<< "add LC between " << prev_global_node_idx << " and " << curr_global_node_idx << std::endl;
                    gtSAMgraph->add(gtsam::BetweenFactor<gtsam::Pose3>(prev_global_node_idx, curr_global_node_idx, relative_pose, robustLoopNoise));
                    writeEdge({prev_node_idx, curr_node_idx}, relative_pose);

                    mKF->lock();
                    // std::cout << "curr_global_node_idx:" << curr_global_node_idx << ", prev_global_node_idx: " << prev_global_node_idx << std::endl;
                    scanMatchStream << keyframeTimes[prev_node_idx] << " " << keyframeTimes[curr_node_idx] << "\n";
                    mKF->unlock();
                    mtxPosegraph->unlock();
                }
            }

            std::chrono::milliseconds dura(2);
            std::this_thread::sleep_for(dura);

            // if (*shutdown)
            //     break;
        }
    }

    void process_inter_icp()
    {
        while (ros::ok())
        {
            // std::cout << "process_inter_icp still running" << std::endl;
            while (!interRobotLoopICPBuf.empty()) {
                if (interRobotLoopICPBuf.size() > 30) {
                    ROS_WARN("Too many inter-robot loop clousre candidates to be ICPed is waiting ... Do process_lcd less frequently (adjust loopClosureFrequency)");
                }

                auto loop_idx_tuple = interRobotLoopICPBuf.front();
                mBuf->lock();
                interRobotLoopICPBuf.pop();
                mBuf->unlock();
                // std::cout << "after_interRobotLoopICPBuf.size(): " << interRobotLoopICPBuf.size() << std::endl;

                const int curr_robot_id = std::get<0>(loop_idx_tuple);
                const int friend_robot_id = std::get<1>(loop_idx_tuple);
                const int curr_node_idx = std::get<2>(loop_idx_tuple);
                const int friend_node_idx = std::get<3>(loop_idx_tuple);

                const int curr_global_node_idx = curr_robot_id * MAX_NODES_PER_ROBOT + curr_node_idx;
                const int friend_global_node_idx = friend_robot_id * MAX_NODES_PER_ROBOT + friend_node_idx;
                
                Robot* curr_robot = this;
                Robot* friend_robot = curr_robot->friend_robots[friend_robot_id].get();

                auto inter_relative_pose_optional = doICPVirtualRelativeFriendRobot(curr_robot, friend_robot, curr_node_idx, friend_node_idx);
   
                if (inter_relative_pose_optional){
                    gtsam::Pose3 inter_relative_pose = inter_relative_pose_optional.value();
                    mtxPosegraph->lock();
                    std::cout << "initialEstimate keys: ";
                    for (const auto& key_value_pair : initialEstimate) {
                        std::cout << key_value_pair.key << " ";
                    }
                    std::cout << std::endl;
                    gtSAMgraph->add(gtsam::BetweenFactor<gtsam::Pose3>(friend_global_node_idx, curr_global_node_idx, inter_relative_pose, robustLoopNoise));
                    mKF->lock();
                    std::cout << "interRobot_relative_pose! " << std::endl;
                    robots_scanMatchStream << curr_robot_id << " " << keyframeTimes[curr_node_idx] << " " <<
                                            friend_robot_id << " " << keyframeTimes[friend_node_idx] << "\n";
                    mKF->unlock();
                    mtxPosegraph->unlock();
                }
            }

            std::chrono::milliseconds dura(2);
            std::this_thread::sleep_for(dura);

            // if (*shutdown)
            //     break;
        }
    }

    void process_viz_path()
    {
        ros::Rate rate(vizPathFrequency);
        while (ros::ok())
        {
            rate.sleep();
            // std::cout << "process_viz_path still running" << std::endl;
            if (laserCloudMapPGORedraw || true)
            {
                pubPath();
            }
            // if (*shutdown)
            //     break;
        }
    }

    void process_isam()
    {
        float hz = 1;
        ros::Rate rate(hz);
        while (ros::ok())
        {
            rate.sleep();
            // std::cout << "process_isam still running for " << robot_id << std::endl;
            // std::cout<<"robot id: "<<robot_id<<std::endl;
            if (*gtSAMgraphMade)  //  && newStateTimes.size() > 0
            {
                // cout << "running isam2 optimization ..." << endl;
                // std::cout<< "keyframePoses.size(): " << keyframePoses.size() << std::endl;
                // std::cout<< "gtSAMgraph->size(): " << gtSAMgraph->size() << std::endl;
                runISAM2opt();
                // std::cout<<"robot id1111: "<<robot_id<<std::endl;
                // std::cout<< "gtSAMgraph->size() aft: " << gtSAMgraph->size() << std::endl;

                // if (*shutdown)
                // {
                //     std::cout << "Saving the posegraph for" << robot_id << " ..." << std::endl;
                //     ROS_INFO_STREAM("Saving odometry poses to file " << odomTUMFormat); 
                //     saveOdometryVerticesTUMFormat(odomTUMFormat, keyframePoses, keyframeTimes, robot_id);
                    
                //     ROS_INFO_STREAM("Saving optimized poses to file " << pgTUMFormat);
                //     saveOptimizedVerticesTUMFormat(isamCurrentEstimate, pgTUMFormat, isamStateTimes, robot_id);
                //     break;
                // }
            }
        }
    }


    gtsam::Pose3 convertPose6DToPose3(const Pose6D& pose6D) {
        gtsam::Point3 translation(pose6D.x, pose6D.y, pose6D.z);
        gtsam::Rot3 rotation = gtsam::Rot3::RzRyRx(pose6D.roll, pose6D.pitch, pose6D.yaw);

        return gtsam::Pose3(rotation, translation);
    }

    std::vector<gtsam::Pose3> convertPose6DVectorToPose3Vector(const std::vector<Pose6D>& pose6DVec) {
        std::vector<gtsam::Pose3> pose3Vec;
        pose3Vec.reserve(pose6DVec.size());

        for (const auto& pose6D : pose6DVec) {
            pose3Vec.push_back(convertPose6DToPose3(pose6D));
        }

        return pose3Vec;
    }

    gtsam::Pose3 interp(const std::vector<gtsam::Pose3>& keyframePosesUpdated, 
                    const std::vector<ros::Time>& keyframeTimes, 
                    ros::Time targetTime) {

        std::cout<< "keyframePosesUpdated.size(): " << keyframePosesUpdated.size() << std::endl;
        std::cout<< "keyframeTimes.size(): " << keyframeTimes.size() << std::endl;

        if (keyframePosesUpdated.size() != keyframeTimes.size()) {
            throw std::runtime_error("Keyframe poses and times size mismatch.");
        }

        // 将 targetTime 转换为 double
        double targetTimeSec = targetTime.toSec();

        // 如果目标时间小于第一个关键帧时间，返回第一个关键帧位姿
        if (targetTimeSec <= keyframeTimes.front().toSec()) {
            return keyframePosesUpdated.front();
        }

        // 如果目标时间大于最后一个关键帧时间，返回最后一个关键帧位姿
        if (targetTimeSec >= keyframeTimes.back().toSec()) {
            return keyframePosesUpdated.back();
        }

        // 查找目标时间位于的两个关键帧时间之间的索引
        size_t idx = 0;
        while (idx < keyframeTimes.size() - 1 && keyframeTimes[idx + 1].toSec() < targetTimeSec) {
            ++idx;
        }

        // 获取对应的两个关键帧时间和位姿，并将其转换为 double
        double t1 = keyframeTimes[idx].toSec();
        double t2 = keyframeTimes[idx + 1].toSec();
        const gtsam::Pose3& pose1 = keyframePosesUpdated[idx];
        const gtsam::Pose3& pose2 = keyframePosesUpdated[idx + 1];

        // 计算插值比例 alpha
        double alpha = (targetTimeSec - t1) / (t2 - t1);

        // 插值平移部分
        gtsam::Point3 translation1 = pose1.translation();
        gtsam::Point3 translation2 = pose2.translation();
        gtsam::Point3 interpTranslation = translation1 * (1.0 - alpha) + translation2 * alpha;

        // 插值旋转部分，使用 SLERP (球面线性插值)
        gtsam::Rot3 rotation1 = pose1.rotation();
        gtsam::Rot3 rotation2 = pose2.rotation();
        gtsam::Rot3 interpRotation = rotation1.slerp(alpha, rotation2);

        // 返回插值后的位姿
        return gtsam::Pose3(interpRotation, interpTranslation);
    }

    void process_dense_pgo()
    {
        float hz = 0.2;
        ros::Rate rate(hz);

        while (ros::ok())
        {
            rate.sleep();  // 控制循环频率

            // 使用 std::lock_guard 自动管理锁的释放
            std::lock_guard<std::mutex> lock(*mtxPosegraph_dense);

            // 初始化 ISAM2 和因子图
            ISAM2Params parameters;
            parameters.relinearizeThreshold = 0.01;
            parameters.relinearizeSkip = 1;
            std::unique_ptr<gtsam::ISAM2> isam_dense = std::make_unique<gtsam::ISAM2>(parameters);
            std::unique_ptr<gtsam::NonlinearFactorGraph> gtSAMgraph_dense = std::make_unique<gtsam::NonlinearFactorGraph>();
            gtsam::Values initialEstimate_dense;

            // 初始化一个新的变量来存储优化后的稠密位姿
            std::vector<gtsam::Pose3> optimizedDensePoses(DenseframePoses.size());

            // 检查 keyframePosesUpdated 和 keyframeTimes 大小是否匹配
            if (keyframePosesUpdated.size() != keyframeTimes.size()) {
                throw std::runtime_error("Keyframe poses and times size mismatch.");
            }

            // 插值 keyframePosesUpdated 到 DenseframePoses 频率
            for (size_t i = 0; i < DenseframePoses.size(); ++i) {
                if (DenseframeTimes[i]>=keyframeTimes[0] && DenseframeTimes[i]<=keyframeTimes[recentIdxUpdated->load() - 1]) {
                    std::vector<gtsam::Pose3> keyframePosesAsPose3 = convertPose6DVectorToPose3Vector(keyframePosesUpdated);
                    optimizedDensePoses[i] = interp(keyframePosesAsPose3, keyframeTimes, DenseframeTimes[i]);
                }
            }

            // 1. 构建因子图 (绝对观测和相对观测)
            for (size_t i = 0; i < DenseframePoses.size() - 1; ++i) {
                // 将 Pose6D 转换为 gtsam::Pose3
                gtsam::Pose3 pose1 = convertPose6DToPose3(DenseframePoses[i]);
                gtsam::Pose3 pose2 = convertPose6DToPose3(DenseframePoses[i + 1]);

                // 添加相对位姿观测
                gtsam::Pose3 T_relative = pose1.between(pose2);
                auto noise_model = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-3, 1e-3, 1e-3).finished());
                gtSAMgraph_dense->add(gtsam::BetweenFactor<gtsam::Pose3>(gtsam::Symbol('x', i), gtsam::Symbol('x', i + 1), T_relative, noise_model));

                // 将 keyframePosesUpdated[i] 从 Pose6D 转换为 gtsam::Pose3
                gtsam::Pose3 keyframePose3 = convertPose6DToPose3(keyframePosesUpdated[i]);

                // 添加绝对位姿观测（基于更新后的关键帧位姿，使用转换后的 gtsam::Pose3）
                auto prior_noise_model = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3).finished());
                gtSAMgraph_dense->add(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('x', i), keyframePose3, prior_noise_model));

                // 添加初始估计
                initialEstimate_dense.insert(gtsam::Symbol('x', i), pose1);
            }

            // 2. 进行 ISAM2 优化
            isam_dense->update(*gtSAMgraph_dense, initialEstimate_dense);
            // 不需要第二次更新
            // isam_dense->update();

            // 3. 获取当前优化结果
            auto isamCurrentEstimate_dense = isam_dense->calculateEstimate();

            // 4. 将优化后的稠密位姿存储到新的变量 optimizedDensePoses 中
            for (size_t i = 0; i < DenseframePoses.size(); ++i) {
                optimizedDensePoses[i] = isamCurrentEstimate_dense.at<gtsam::Pose3>(gtsam::Symbol('x', i));
            }

            // 优化完成后，保存新的稠密位姿
            saveOdometryVerticesTUMFormat(denseTUMFormat, DenseframePoses, keyframeTimes, robot_id);

            // 5. 清理因子图和初始估计，以便下次迭代
            gtSAMgraph_dense->resize(0);
            initialEstimate_dense.clear();
        }
    }


    // void process_dense_pgo()
    // {
    //     float hz = 0.2;
    //     ros::Rate rate(hz);
    //     while (ros::ok())
    //     {
    //         rate.sleep();  // 控制循环频率
            
    //         // std::shared_ptr<std::mutex> mtxPosegraph_dense(new std::mutex()); // TODO yiwen: defined as class member
    //         // std::lock_guard<std::mutex> lock(*mtxPosegraph_dense);  // 锁定互斥锁，确保线程安全 

    //         // 初始化 ISAM2 和因子图
    //         ISAM2Params parameters;
    //         parameters.relinearizeThreshold = 0.01;
    //         parameters.relinearizeSkip = 1;
    //         gtsam::ISAM2 *isam_dense = new gtsam::ISAM2(parameters);
    //         gtsam::NonlinearFactorGraph* gtSAMgraph_dense = new gtsam::NonlinearFactorGraph();
    //         gtsam::Values initialEstimate_dense;

    //         // 初始化一个新的变量来存储优化后的稠密位姿
    //         mtxPosegraph_dense->lock();
    //         std::vector<gtsam::Pose3> optimizedDensePoses;
    //         optimizedDensePoses.resize(DenseframePoses.size());
    //         // interp keyframePosesUpdated to DenseframePoses freq

    //         for(size_t i = 0; i < DenseframePoses.size(); ++i) {
    //             if (DenseframeTimes[i]>=keyframeTimes[0] && DenseframeTimes[i]<=keyframeTimes[recentIdxUpdated->load() - 1]) {
    //                 std::vector<gtsam::Pose3> keyframePosesAsPose3 = convertPose6DVectorToPose3Vector(keyframePosesUpdated);
    //                 optimizedDensePoses[i] = interp(keyframePosesAsPose3, keyframeTimes, DenseframeTimes[i]); // TODO yiwen: interp keyframePosesUpdated to DenseframePoses freq
    //             }
    //         }

    //         // 1. 构建因子图 (绝对观测和相对观测)
    //         for (size_t i = 0; i < DenseframePoses.size() - 1; ++i) {
    //             // 将 Pose6D 转换为 gtsam::Pose3
    //             gtsam::Pose3 pose1 = convertPose6DToPose3(DenseframePoses[i]);
    //             gtsam::Pose3 pose2 = convertPose6DToPose3(DenseframePoses[i + 1]);

    //             // 添加相对位姿观测
    //             gtsam::Pose3 T_relative = pose1.between(pose2);
    //             auto noise_model = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-3, 1e-3, 1e-3).finished());
    //             gtSAMgraph_dense->add(gtsam::BetweenFactor<gtsam::Pose3>(gtsam::Symbol('x', i), gtsam::Symbol('x', i + 1), T_relative, noise_model));

    //             // 将 keyframePosesUpdated[i] 从 Pose6D 转换为 gtsam::Pose3
    //             gtsam::Pose3 keyframePose3 = convertPose6DToPose3(keyframePosesUpdated[i]);

    //             // 添加绝对位姿观测（基于更新后的关键帧位姿，使用转换后的 gtsam::Pose3）
    //             auto prior_noise_model = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3).finished());
    //             gtSAMgraph_dense->add(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('x', i), keyframePose3, prior_noise_model));

    //             // 添加初始估计
    //             initialEstimate_dense.insert(gtsam::Symbol('x', i), pose1);
    //         }
    //         mtxPosegraph_dense->unlock();

    //     //     // 2. 进行 ISAM2 优化
    //         isam_dense->update(*gtSAMgraph_dense, initialEstimate_dense);
    //         isam_dense->update();  // 执行增量更新

    //         // 3. 获取当前优化结果
    //         auto isamCurrentEstimate_dense = isam_dense->calculateEstimate();

    //         // 4. 将优化后的稠密位姿存储到新的变量 optimizedDensePoses 中
    //         for (size_t i = 0; i < DenseframePoses.size(); ++i) {
    //             optimizedDensePoses[i] = isamCurrentEstimate_dense.at<gtsam::Pose3>(gtsam::Symbol('x', i));
    //         }

    //         // 优化完成后，保存新的稠密位姿
    //         saveOdometryVerticesTUMFormat(denseTUMFormat, DenseframePoses, keyframeTimes, robot_id);

    //     //     // 5. 清理因子图和初始估计，以便下次迭代
    //         gtSAMgraph_dense->resize(0);
    //         initialEstimate_dense.clear();
    //     }
    // }

    void pubMap()
    {
        int SKIP_FRAMES = 2;
        int counter = 0;

        laserCloudMapPGO->clear();

        // std::cout<< "recentIdxUpdated: " << *recentIdxUpdated <<std::endl;

        mKF->lock();
        for (int node_idx = 0; node_idx < *recentIdxUpdated; node_idx++)
        // for (int node_idx = 0; node_idx < keyframePoses.size(); node_idx++)
        {
            if (counter % SKIP_FRAMES == 0)
            {
                // std::cout<< "map pc size: " << laserCloudMapPGO->size() << std::endl;
                *laserCloudMapPGO += *local2global(keyframeLaserClouds[node_idx], keyframePosesUpdated[node_idx]);
            }
            counter++;
        }
        mKF->unlock();

        downSizeFilterMapPGO.setInputCloud(laserCloudMapPGO);
        downSizeFilterMapPGO.filter(*laserCloudMapPGO);

        sensor_msgs::PointCloud2 laserCloudMapPGOMsg;
        pcl::toROSMsg(*laserCloudMapPGO, laserCloudMapPGOMsg);
        laserCloudMapPGOMsg.header.frame_id = "camera_init";
        pubMapAftPGO.publish(laserCloudMapPGOMsg);
    }

    void process_viz_map()
    {
        // ros::Rate rate(vizmapFrequency);
        ros::Rate rate(1);
        while (ros::ok())
        {
            rate.sleep();
            // std::cout << "process_viz_map still running" << std::endl;
            if (laserCloudMapPGORedraw || true)
            {
                pubMap();
            }
            // if (*shutdown)
            //     break;
        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "multi_robot_slam");
    ros::NodeHandle nh;

    // auto gtSAMgraph_global = new gtsam::NonlinearFactorGraph();
    // gtsam::NonlinearFactorGraph gtSAMgraph_global;
    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    gtsam::ISAM2 *isam_global = new ISAM2(parameters);
    std::shared_ptr<std::mutex> mtxPosegraph_global(new std::mutex());

    int num_robots = 3; // 假设有三个机器人

    std::vector<std::shared_ptr<Robot>> robots;
    for (int i = 0; i < num_robots; ++i)
    {
        robots.emplace_back(std::make_shared<Robot>(nh, i, isam_global, mtxPosegraph_global));
    }

    // 设置friend_robots数组
    for (int i = 0; i < num_robots; ++i)
    {
        for (int j = 0; j < num_robots; ++j)
        {
            if (i != j)
            {
                robots[i]->friend_robots[j] = robots[j];
            }
        }
    }

    // 初始化订阅和发布话题
    std::vector<ros::Subscriber> odometry_subscribers;
    std::vector<ros::Subscriber> fullRes_subscribers;
    std::vector<ros::Subscriber> gps_subscribers;

    for (int i = 0; i < num_robots; ++i)
    {
        odometry_subscribers.push_back(nh.subscribe<nav_msgs::Odometry>(
            "/robot_" + std::to_string(i) + "/odom", 100, 
            [i, &robots](const nav_msgs::Odometry::ConstPtr& msg) { robots[i]->laserOdometryHandler(msg); }
        ));
        fullRes_subscribers.push_back(nh.subscribe<sensor_msgs::PointCloud2>(
            "/robot_" + std::to_string(i) + "/full_cloud", 100, 
            [i, &robots](const sensor_msgs::PointCloud2ConstPtr& msg) { robots[i]->laserCloudFullResHandler(msg); }
        ));
        gps_subscribers.push_back(nh.subscribe<sensor_msgs::NavSatFix>(
            "/robot_" + std::to_string(i) + "/gps", 100, 
            [i, &robots](const sensor_msgs::NavSatFix::ConstPtr& msg) { robots[i]->gpsHandler(msg); }
        ));
    }

    // 启动处理线程
    std::vector<std::thread> processing_threads;
    for (auto &robot : robots)
    {
        processing_threads.emplace_back(&Robot::process_pg, robot);
        processing_threads.emplace_back(&Robot::process_lcd, robot); // loop closure detect
        processing_threads.emplace_back(&Robot::process_icp, robot); // icp pose get (for lc)
        processing_threads.emplace_back(&Robot::process_inter_icp, robot); // icp pose get (for inter lc)
        processing_threads.emplace_back(&Robot::process_isam, robot);
        // processing_threads.emplace_back(&Robot::process_dense_pgo, robot); // process_dense_pgo
        processing_threads.emplace_back(&Robot::process_viz_path, robot);
        processing_threads.emplace_back(&Robot::process_viz_map, robot);
    }

    ros::spin();

    for (auto &thread : processing_threads)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }

    return 0;
}
