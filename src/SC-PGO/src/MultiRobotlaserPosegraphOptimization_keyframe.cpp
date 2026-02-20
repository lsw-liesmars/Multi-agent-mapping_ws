// // 只存储使用关键帧
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
#include <condition_variable>

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
    // std::queue<std::tuple<int, int, int, int>> allinterRobotLoopICPBuf;

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

    bool *inter_init;
    std::vector<std::tuple<int, int, gtsam::Pose3>> inter_loop_container;
    
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
    std::string saveNogroundDirectory;
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
    int last_lcd_size = 0;

    int robot_id;
    std::unordered_map<int, std::shared_ptr<Robot>> friend_robots;
    int index_offset;

    Eigen::Affine3f scInitialGuess;
    std::mutex scGuessMutex;

    // Quatro
    bool useQuatro;
    int useQuatroCount = 0;
    double voxel_size, normal_radius, fpfh_radius;
    bool estimating_scale;
    double noise_bound, noise_bound_coeff, gnc_factor, rot_cost_diff_thr;
    int num_max_iter;
    std::string lidarType, groundSegMode, neighborSelectionMode;
    std::mutex matchMutex; 

    Quatro<PointType, PointType>::Params quatroParams;
    CSFParams csf_params;

    ros::NodeHandle nh;

    // void initializeQuatroParams(ros::NodeHandle &nh);

    Robot(ros::NodeHandle &nh, int id, gtsam::ISAM2 *isam_global, std::shared_ptr<std::mutex> mtxPosegraph_global, 
        bool *inter_init_global, std::vector<std::tuple<int, int, gtsam::Pose3>> &inter_loop_container_global):
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
        inter_init = inter_init_global;
        inter_loop_container = inter_loop_container_global;
    }

    void initializeQuatroParams(ros::NodeHandle &nh) 
    {
        nh.param<bool>("use_quatro", useQuatro, false);
        nh.param<bool>("filter_ground", filterGround, true);
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

        pgTUMFormat = save_directory + "robot_" + std::to_string(id)+"_keyscan_optimized_poses.txt";
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

        saveNogroundDirectory = save_directory + "NogroundSubmaps/";
        unused = system((std::string("mkdir -p ") + saveNogroundDirectory).c_str());
        unused = system((std::string("exec rm -r ") + saveNogroundDirectory).c_str());
        unused = system((std::string("mkdir -p ") + saveNogroundDirectory).c_str());

        nh.param<double>("keyframe_meter_gap", keyframeMeterGap, 1.0);
        nh.param<double>("keyframe_deg_gap", keyframeDegGap, 2.0);
        // keyframeRadGap = deg2rad(keyframeDegGap);
        keyframeRadGap = keyframeDegGap * M_PI / 180.0;


        // loop closure detection parameter setting
        nh.param<int>("lcd_mode", lcd_mode, 0);
        nh.param<int>("halfsubmapsize", halfsubmapsize, 5);
        nh.param<std::string>("deepmodelpath", deepmodelpath, "/");
        last_frame_sub_time = ros::Time::now().toSec();

        if(lcd_mode==1){
            nh.param<double>("sc_dist_thres", scDistThres, 0.2);
            nh.param<double>("Multi_Agents_SC_DIST_THRES", Multi_scDistThres, 0.2);
            nh.param<double>("sc_max_radius", scMaximumRadius, 80.0);
        }
        else if(lcd_mode==2){ // 
            // std::string deepmodelpath = "/home/wbl/code/kuangshan/deep_reloc/PointNetVlad-best-1018.pt";
            nh.param<int>("des_dim", deepDescManager.outputDim_, 256);
            deepDescManager.setParams(deepmodelpath, 4096, 3, deepDescManager.outputDim_); // std::unique_ptr cannot use "=" to copy
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
        // odomNoiseVector6 << 1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2;
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
        // if (newStateTimes.size() != initialEstimate.size()) {
        //     ROS_ERROR("Inconsistent new state times and initial estimates.");
        // }
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

    void loopFindNearframesCloud_DS( pcl::PointCloud<PointType>::Ptr& local_nearframes, pcl::PointCloud<PointType>::Ptr& local_nearframes_DS,
                                     const int& key, const int& submap_size, const int& root_idx)
    {
        // extract and stacking near keyframes (in global coord)
        pcl::PointCloud<PointType>::Ptr nearframes(new pcl::PointCloud<PointType>());
        // Eigen::Vector3d transPos = Eigen::Vector3d(0,0,0);
        // transPos << allframePoses[root_idx].x, allframePoses[root_idx].y, allframePoses[root_idx].z;
        for (int i = -submap_size; i <= submap_size; ++i) {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= int(allframeLaserClouds.size()) ){
                // std::cout << "keyNear: err!! " <<keyNear<< std::endl; 
                continue;
            }

            // odometry T_B1_B2
            // *nearKeyframes += * local2global(keyframeLaserClouds[keyNear], keyframePosesUpdated[root_idx]);
            *nearframes += * local2global(allframeLaserClouds[keyNear], allframePoses[keyNear]); // to global                                                                   keyframePosesUpdated[key]); // to local
        }

        if (nearframes->empty())
            return;

        // Remove NaN points
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*nearframes, *nearframes, indices);

        // add by yiwen
        const Pose6D W_T_B = allframePoses[root_idx];
        Eigen::Affine3f affine_W_T_B = pcl::getTransformation(W_T_B.x, W_T_B.y, W_T_B.z, W_T_B.roll, W_T_B.pitch, W_T_B.yaw);
        gtsam::Pose3 gtsam_W_T_B(gtsam::Rot3(affine_W_T_B.rotation().cast<double>()), affine_W_T_B.translation().cast<double>());
        gtsam::Pose3 gtsam_B_T_W = gtsam_W_T_B.inverse();
        // local_nearframes = transformPointCloud(nearframes, gtsam_B_T_W);
        pcl::PointCloud<PointType>::Ptr transformedNearKeyframes = transformPointCloud(nearframes, gtsam_B_T_W);

        // Downsample the point cloud
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        {
            // std::scoped_lock lock(*downICPFilterMutex);
            std::lock_guard<std::mutex> lock(*downICPFilterMutex);
            downSizeFilterICP.setInputCloud(transformedNearKeyframes);
            downSizeFilterICP.filter(*cloud_temp);

        }
        *local_nearframes = *transformedNearKeyframes;
        *local_nearframes_DS = *cloud_temp;
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

        *local_nearframes = *nearframes;
    }

       void loopFindNearKeyframesCloud2(pcl::PointCloud<PointType>::Ptr &local_nearframes, const int &key, const int &submap_size, const int &root_idx, const size_t min_points = 1000)
    {
        // extract and stacking near keyframes (in global coord)
        pcl::PointCloud<PointType>::Ptr nearframes(new pcl::PointCloud<PointType>());
    
        // 记录已添加的帧索引
        std::unordered_set<int> added_frames;
    
        for (int i = -submap_size; i <= submap_size; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= int(keyframePoses.size()))
            {
                continue;
            }
    
            *nearframes += *local2global(keyframeLaserClouds[keyNear], keyframePoses[keyNear]);
            added_frames.insert(keyNear);
        }
    
        // 如果点云为空，直接返回
        if (nearframes->empty())
            return;
    
    
        // 检查点云数量是否满足预期
        int additional_frames = 1; // 每次增加的近邻帧数量
        while (nearframes->size() < min_points)
        {
            // 增加更远的近邻帧
            for (int i = -submap_size - additional_frames; i <= submap_size + additional_frames; ++i)
            {
                int keyNear = key + i;
                if (keyNear < 0 || keyNear >= int(keyframePoses.size()))
                {
                    continue;
                }
    
                // 跳过已添加的帧
                if (added_frames.find(keyNear) != added_frames.end())
                {
                    continue;
                }
    
                // 添加帧到点云，并记录索引
                *nearframes += *local2global(keyframeLaserClouds[keyNear], keyframePoses[keyNear]);
                added_frames.insert(keyNear);
            }
    
            // 增加搜索范围
            additional_frames++;
    
            // 如果搜索范围已经覆盖所有关键帧，退出循环
            if (key - submap_size - additional_frames < 0 && key + submap_size + additional_frames >= int(keyframePoses.size()))
            {
                break;
            }
        }
    
        // 将点云转换到局部坐标系
        const Pose6D W_T_B = keyframePoses[root_idx];
        Eigen::Affine3f affine_W_T_B = pcl::getTransformation(W_T_B.x, W_T_B.y, W_T_B.z, W_T_B.roll, W_T_B.pitch, W_T_B.yaw);
        gtsam::Pose3 gtsam_W_T_B(gtsam::Rot3(affine_W_T_B.rotation().cast<double>()), affine_W_T_B.translation().cast<double>());
        gtsam::Pose3 gtsam_B_T_W = gtsam_W_T_B.inverse();
        local_nearframes = transformPointCloud(nearframes, gtsam_B_T_W);
    }
    void loopFindNearframesCloud2( pcl::PointCloud<PointType>::Ptr& local_nearframes, const int& key, const int& submap_size, const int& root_idx)
    {
        // extract and stacking near keyframes (in global coord)
        pcl::PointCloud<PointType>::Ptr nearframes(new pcl::PointCloud<PointType>());
        // Eigen::Vector3d transPos = Eigen::Vector3d(0,0,0);
        // transPos << allframePoses[root_idx].x, allframePoses[root_idx].y, allframePoses[root_idx].z;
        for (int i = -submap_size; i <= submap_size; ++i) {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= int(allframeLaserClouds.size()) ){
                // std::cout << "keyNear: err!! " <<keyNear<< std::endl; 
                continue;
            }

            // odometry T_B1_B2
            // *nearKeyframes += * local2global(keyframeLaserClouds[keyNear], keyframePosesUpdated[root_idx]);
            *nearframes += * local2global(allframeLaserClouds[keyNear], allframePoses[keyNear]); // to global                                                                   keyframePosesUpdated[key]); // to local
        }

        if (nearframes->empty())
            return;

        // Remove NaN points
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*nearframes, *nearframes, indices);

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
        std::cout << "cureKeyframeCloud_DS size: " << cureKeyframeCloud_DS->size() << std::endl;

        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(2 * historyKeyframeSearchNum * keyframeMeterGap);
        icp.setMaximumIterations(50);
        icp.setTransformationEpsilon(1e-6);
        // icp.setEuclideanFitnessEpsilon(1e-1);

        icp.setInputSource(cureKeyframeCloud_DS);
        icp.setInputTarget(targetKeyframeCloud_DS);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        mKF->lock();
        // Pose6D loopKfPose = keyframePosesUpdated[_loop_kf_idx];
        // Eigen::Affine3f guess = toEigenAffine3f(loopKfPose).inverse() * toEigenAffine3f(keyframePosesUpdated[_curr_kf_idx]);
        Pose6D loopKfPose = keyframePoses[_loop_kf_idx];
        Eigen::Affine3f guess = toEigenAffine3f(loopKfPose).inverse() * toEigenAffine3f(keyframePoses[_curr_kf_idx]);
        guess.translation() = Eigen::Vector3f(0, 0, 0);

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

    void logRotationAngleAndRobots(int cur_global_idx, int friend_global_idx, const Eigen::Matrix4f& initial_guess, 
                                    const Eigen::Matrix4f& icp_trans, double icp_fitness_score, std::string save_path)
    {
        Eigen::Matrix3f rotation_matrix = initial_guess.block<3, 3>(0, 0);
        Eigen::Vector3f euler_angles = rotation_matrix.eulerAngles(2, 1, 0) * 180.0 / M_PI;
        // float yaw_angle = std::atan2(rotation_matrix(1, 0), rotation_matrix(0, 0)) * 180.0 / M_PI;

        Eigen::Matrix3f icp_rotation_matrix = icp_trans.block<3, 3>(0, 0);
        Eigen::Vector3f icp_euler_angles = icp_rotation_matrix.eulerAngles(2, 1, 0) * 180.0 / M_PI;
        // float icp_yaw_angle = std::atan2(icp_rotation_matrix(1, 0), icp_rotation_matrix(0, 0)) * 180.0 / M_PI;
        

        std::ofstream outfile;
        outfile.open(save_path, std::ios_base::app);

        // outfile << "cur_global_idx:" << cur_global_idx<< ", friend_global_idx:" << friend_global_idx
        //         << ", Initial Guess Rotation (yaw in radians): " << yaw_angle << " ,yaw_icp: " << icp_yaw_angle <<
        //         ", ICP Fitness Score: " << icp_fitness_score << "\n";
        // outfile << "cur_global_idx:" << cur_global_idx<< ", friend_global_idx:" << friend_global_idx
        //         << ", Initial Guess Rotation (roll, pitch, yaw in degrees): " << euler_angles.transpose() << " ,yaw_icp: " << icp_euler_angles.transpose() <<
        //         ", ICP Fitness Score: " << icp_fitness_score << "\n";

        outfile << "cur_global_idx:" << cur_global_idx<< ", friend_global_idx:" << friend_global_idx
                << ", Initial Guess Rotation : \n" << rotation_matrix <<
                "\n ICP Rotation: \n" << icp_rotation_matrix << "\n";
        
        outfile.close();
    }

    
    bool isHighCandidateRegion(const std::queue<std::tuple<int, int, int, int>>& interLoopBuf, 
                           int curr_node_idx, int window_size = 10, float threshold = 0.6) {
        std::deque<std::tuple<int, int, int, int>> temp_queue;
        
        std::queue<std::tuple<int, int, int, int>> queue_copy = interLoopBuf;
        while (!queue_copy.empty()) {
            temp_queue.push_back(queue_copy.front());
            queue_copy.pop();
        }

        int count = 0;
        int total_checked = 0;

        for (auto it = temp_queue.rbegin(); it != temp_queue.rend() && total_checked < window_size; ++it) {
            int candidate_idx = std::get<2>(*it); 
            if (candidate_idx >= curr_node_idx - window_size && candidate_idx < curr_node_idx) {
                count++; 
            }
            total_checked++;
        }

        float match_ratio = static_cast<float>(count) / window_size;
        return match_ratio >= threshold;
    }

    std::optional<gtsam::Pose3> doICPVirtualRelativeFriendRobot(Robot* curr_robot, Robot* friend_robot, int curr_node_idx, int friend_node_idx)
    {
        // 确保关键帧索引有效
        if (curr_node_idx >= curr_robot->keyframePoses.size() || friend_node_idx >= friend_robot->keyframePoses.size()) {
            ROS_ERROR("Node index out of bounds");
            return std::nullopt;
        }

        // 初始化点云
        pcl::PointCloud<PointType>::Ptr currKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr currKeyframeCloud_DS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr friendKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr friendKeyframeCloud_DS(new pcl::PointCloud<PointType>());

        curr_robot->loopFindNearKeyframesCloud(currKeyframeCloud, currKeyframeCloud_DS, 
                                            curr_node_idx, historyKeyframeSearchNum, curr_node_idx);
        friend_robot->loopFindNearKeyframesCloud(friendKeyframeCloud, friendKeyframeCloud_DS, 
                                                friend_node_idx, historyKeyframeSearchNum, friend_node_idx);

        // curr_robot->loopFindNearframesCloud_DS(currKeyframeCloud, currKeyframeCloud_DS, curr_robot->keyframeIndexTOallID[curr_node_idx], 
        //                                 historyKeyframeSearchNum, curr_robot->keyframeIndexTOallID[curr_node_idx]);
        // friend_robot->loopFindNearframesCloud_DS(friendKeyframeCloud, friendKeyframeCloud_DS, friend_robot->keyframeIndexTOallID[friend_node_idx], 
        //                                 historyKeyframeSearchNum, friend_robot->keyframeIndexTOallID[friend_node_idx]);

        // 全局索引
        int cur_global_idx = curr_node_idx + curr_robot->index_offset;
        int friend_global_idx = friend_node_idx + friend_robot->index_offset;

        // 移除NaN点
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*currKeyframeCloud_DS, *currKeyframeCloud_DS, indices);
        pcl::removeNaNFromPointCloud(*friendKeyframeCloud_DS, *friendKeyframeCloud_DS, indices);

        // ICP配置
        pcl::IterativeClosestPoint<PointType, PointType> icp;
        // pcl::GeneralizedIterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(2 * historyKeyframeSearchNum * curr_robot->keyframeMeterGap);
        // icp.setMaxCorrespondenceDistance(100.0);
        icp.setMaximumIterations(50);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setInputSource(currKeyframeCloud_DS);
        icp.setInputTarget(friendKeyframeCloud_DS);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());

        Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();

        bool choose_quatro = false;

        // 关键帧计数逻辑
        if (useQuatro) {
        // if (useQuatro && (curr_robot->keyframePoses.size() % 5 == 0 && curr_robot->useQuatroCount < 5)) {
            // std::cout << "curr_robot:" << this->robot_id << ", curr_node_idx:" << curr_node_idx << std::endl;
            // std::cout << "robot-useQuatroCount: " << this->robot_id<< "-" << curr_robot->useQuatroCount << std::endl;
            // 使用Quatro进行初值计算
            std::lock_guard<std::mutex> lock(matchMutex);
            Eigen::Matrix4d quatro_output;

            pcl::PointCloud<PointType>::Ptr cur_nonground(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr friend_noground(new pcl::PointCloud<PointType>());

            ros::NodeHandle &nh = this->nh;

            matchPointClouds(nh, currKeyframeCloud, friendKeyframeCloud, quatroParams, lidarType, neighborSelectionMode, 
                            groundSegMode, voxel_size, normal_radius, fpfh_radius, quatro_output, csf_params, 
                            cur_nonground, friend_noground, filterGround);
            
            // matchPointClouds(currKeyframeCloud, friendKeyframeCloud, quatroParams, lidarType, neighborSelectionMode, 
                            // groundSegMode, voxel_size, normal_radius, fpfh_radius, quatro_output, csf_params);
            
            initial_guess = quatro_output.cast<float>();

            pcl::io::savePCDFileBinary(saveNogroundDirectory + padZeros(cur_global_idx) + ".pcd", *cur_nonground);
            pcl::io::savePCDFileBinary(saveNogroundDirectory + padZeros(friend_global_idx) + ".pcd", *friend_noground);
            
            choose_quatro = true; 
        }

        // if (initial_guess.isZero(0)) {
        //     std::cout << "Invalid transformation. Skipping loop closure."  << std::endl;
        //     return std::nullopt;
        // }

        // if (choose_quatro){
        //     std::string save_path = "/home/cyw/temp/used_quatro.txt";
        //     logRotationAngleAndRobots(cur_global_idx, friend_global_idx, initial_guess, icp.getFinalTransformation(), icp.getFitnessScore(), save_path);
        // }

        icp.align(*unused_result, initial_guess);
        float rotang = diffRotation(initial_guess.block<3, 3>(0, 0), icp.getFinalTransformation().block<3, 3>(0, 0));

        // std::cout <<"icp.getFitnessScore(): " << icp.getFitnessScore() << "between " << cur_global_idx << " and " << friend_global_idx << std::endl;

        if (!icp.hasConverged() || icp.getFitnessScore() > curr_robot->Multi_loopFitnessScoreThreshold) {
            if (choose_quatro){
                std::string save_path = "/home/cyw/temp/failed_quatro.txt";
                logRotationAngleAndRobots(cur_global_idx, friend_global_idx, initial_guess, icp.getFinalTransformation(), icp.getFitnessScore(), save_path);
            }
            return std::nullopt;
        } else {
            ROS_INFO_STREAM("[ loop "<<cur_global_idx<<"-"<<friend_global_idx<<"] ICP fitness test passed (" 
                            << icp.getFitnessScore() << " < " << curr_robot->Multi_loopFitnessScoreThreshold
                            << ", rot angle change " << rotang * 180 / M_PI << "). Add this multi agent loop.");
            {
                std::lock_guard<std::mutex> lock(*curr_robot->interloopPairMutex);
                curr_robot->interRobotLoopIndexContainer[curr_node_idx] = std::make_pair(friend_robot->robot_id, friend_node_idx);
            }
            std::string save_path = "/home/cyw/temp/successed_res.txt";
            // if(choose_quatro){
                logRotationAngleAndRobots(cur_global_idx, friend_global_idx, initial_guess, icp.getFinalTransformation(), icp.getFitnessScore(), save_path);
            // }
        }

        if (choose_quatro) {
            std::cout << "Using Quatro and success for initial guess" << std::endl;
            useQuatroCount++;
            std::cout << "initial_guess: " << initial_guess << std::endl;
        }
        
        Eigen::Affine3f correctionLidarFrame(icp.getFinalTransformation());
        Eigen::Matrix3f rot = correctionLidarFrame.rotation();
        Eigen::Vector3f trans = correctionLidarFrame.translation();

        return gtsam::Pose3(gtsam::Rot3(rot.cast<double>()), trans.cast<double>());  // return relative transform in lidar frame
    }

    void process_pg()
    {
        while (1)
        {
            // std::cout<<"process_pg still running"<<std::endl;
            while (!odometryBuf.empty() && !fullResBuf.empty())
            {   
                // std::cout<< odometryBuf.size() <<" " <<fullResBuf.size()<<std::endl;
                // std::lock_guard<std::mutex> lock(*mBuf);
                mBuf->lock();
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

                // std::cout << "odometryBuf:" << odometryBuf.size() << std::endl;
                // std::cout << "fullResBuf_size: " << fullResBuf.size() << std::endl;

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
                mBuf->unlock();

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

                // allframeLaserClouds.push_back(thisKeyFrame);
                // allframePoses.push_back(pose_curr);

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

                // if(robot_id == 0){
                //     std::cout << "keyframe added, robot 0." <<keyframePoses.size()<< std::endl;
                // }

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
                // keyframeIndexTOallID.insert(std::pair<int, int>(keyframePoses.size()-1, allframeLaserClouds.size()-1));

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
                        loopFindNearKeyframesCloud2(thisKeySubmap, cur_submap_index, halfsubmapsize, cur_submap_index); 
                        // loopFindNearframesCloud2(thisKeySubmap, keyframeIndexTOallID[cur_submap_index], halfsubmapsize, keyframeIndexTOallID[cur_submap_index]);
                        
                        t_make_submap.toc("make submap");

                        // std::chrono::system_clock::time_point des_start_time = std::chrono::system_clock::now();

                        // todo yiwen: save thisKeySubmap in vector for add_deep_desc.
                        // desc_pcd_vec.push_back(*thisKeySubmap);
                   
                        DeepDescMutex->lock();
                        std::cout << "thisKeySubmap size: " << thisKeySubmap->size() << std::endl;
                        // std::cout << "thisKeySubmap: " << thisKeySubmap->points[0].x << ", " << thisKeySubmap->points[0].y << ", " << thisKeySubmap->points[0].z << std::endl;
                        deepDescManager.makeAndSaveDeepDescAndKeys(*thisKeySubmap);
                        DeepDescMutex->unlock();

                        // std::chrono::system_clock::time_point des_end_time = std::chrono::system_clock::now();
                        // std::chrono::duration<double> des_save_sec = des_end_time - des_start_time; 
                        // std::cout << "des_save_sec: " << des_save_sec.count() << std::endl;

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
                            //     T << 0.0, -1, 0, 0.0000000,
                            //     1,  0.0, 0.000000000000, 0,
                            //     0.000000000000, 0.000000000000, 1.000000000000, 0.0,
                            //     0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000;}
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


    void add_reloc_desc(){
        ros::Rate rate(100);
        while (ros::ok())
        {
            rate.sleep();
            // std::cout << "process_lcd still running" << std::endl;
            if (lcd_mode == 1){
                // add sc
            }
            else if (lcd_mode == 2){
                // add deep_desc
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
        
        if (last_lcd_size == scManager.polarcontext_invkeys_mat_.size()){
            return;
        }
        else{
            last_lcd_size = scManager.polarcontext_invkeys_mat_.size();
        }

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

        if (last_lcd_size == deepDescManager.deepdesc_invkeys_mat_.size()){
            return;
        }
        else{
            last_lcd_size = deepDescManager.deepdesc_invkeys_mat_.size();
        }
        
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
            

            // space diff judge
            if (sqrt((keyframePoses[prev_node_idx].x - keyframePoses[curr_node_idx].x)*(keyframePoses[prev_node_idx].x - keyframePoses[curr_node_idx].x) + 
                    (keyframePoses[prev_node_idx].y - keyframePoses[curr_node_idx].y)*(keyframePoses[prev_node_idx].y - keyframePoses[curr_node_idx].y) + 
                    (keyframePoses[prev_node_idx].z - keyframePoses[curr_node_idx].z)*(keyframePoses[prev_node_idx].z - keyframePoses[curr_node_idx].z)) > historyKeyframeSearchRadius)
            {
                // cout << "Loop invalid becaues space diff large." << endl;
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

        for (auto& [friend_robot_id, friend_robot] : friend_robots) {
            if(std::try_lock(*scMutex, *friend_robot->scMutex) == -1){
                auto detectResult = detectSCLoopIDFriendRobot(*this, friend_robot);
                friend_robot->scMutex->unlock();
                scMutex->unlock();

                int curr_robot_id = std::get<0>(detectResult);
                // int friend_robot_id = std::get<1>(detectResult);
                int loop_id = std::get<2>(detectResult);
                float yaw_diff_rad = std::get<3>(detectResult);

                const int curr_node_idx = this->scManager.polarcontext_invkeys_mat_.size() - 1;
                const int friend_node_idx = loop_id;
                // if (curr_robot_id == 0){
                //     // std::cout << "numPoses_id:" << numPoses - 1 <<std::endl;
                //     std::cout << "curr_node_idx:" << curr_node_idx << std::endl;
                // }


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
                    // allinterRobotLoopICPBuf.push(std::make_tuple(curr_robot_id, friend_robot_id, curr_node_idx, friend_node_idx));
                    mBuf->unlock();
                }
            }
        }
    }

    std::tuple<int, int, int, float> detectSCLoopIDFriendRobot(Robot& robot, std::shared_ptr<Robot>& friend_robot) {
        int loop_id = -1;
        int friend_robot_idx = -1;
        float yaw_diff_rad = 0.0;
        int nn_idx = 0;

        if (friend_robot->scManager.polarcontext_invkeys_mat_.empty() || friend_robot->scManager.polarcontext_invkeys_mat_[0].empty()) {
            ROS_WARN("polarcontext_invkeys_mat_ is empty for friend_robot_id: %d", friend_robot->robot_id);
            return std::make_tuple(robot.robot_id, friend_robot->robot_id, loop_id, yaw_diff_rad);
        }

        if (robot.robot_id==0){
            std::cout << "tree_size:" << friend_robot->scManager.polarcontext_invkeys_to_search_.size() << std::endl;
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

        if(last_inter_lcd_size==robot.scManager.polarcontext_invkeys_mat_.size()){
            return std::make_tuple(robot.robot_id, friend_robot_idx, loop_id, yaw_diff_rad);
        }

        last_inter_lcd_size = robot.scManager.polarcontext_invkeys_mat_.size();

        auto curr_key = robot.scManager.polarcontext_invkeys_mat_.back();
        auto curr_desc = robot.scManager.polarcontexts_.back();

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
                nn_idx = candidate_indexes[candidate_iter_idx];
                friend_robot_idx = friend_robot->robot_id;
            }
        }
        // if (robot.robot_id == 2 || friend_robot_idx == 2) {
        //             std::cout << "robot_id: " << robot.robot_id << " friend_robot_id: " << friend_robot->robot_id << " loop_id: " << loop_id << " min_dist: " << min_dist << std::endl;
        //         }


        std::string save_path = "/home/cyw/temp/SC_loop.txt";
        if (robot.robot_id == 0){ 
            std::ofstream outfile;
            outfile.open(save_path, std::ios_base::app);
            outfile << "robot_id:" << robot.robot_id << " curr_key_idx:" << robot.scManager.polarcontext_invkeys_mat_.size() - 1 << "\n";
            outfile.close();
        }

        if (min_dist < Multi_scDistThres) {
            loop_id = nn_idx;
            std::cout << "robot_id: " << robot.robot_id << " friend_robot_id: " << friend_robot->robot_id << " loop_id: " << loop_id << " min_dist: " << min_dist << std::endl;
            return std::make_tuple(robot.robot_id, friend_robot_idx, loop_id, yaw_diff_rad);
        }

        return std::make_tuple(robot.robot_id, friend_robot_idx, loop_id, yaw_diff_rad);
    }

    void performFriendDeepDescLoopClosure() {
        mKF->lock();
        int numPoses = int(keyframePoses.size());
        mKF->unlock();

        // if(robot_id==0){
        //     return;
        // }

        for (auto& [friend_robot_id, friend_robot] : friend_robots) {
            if (std::try_lock(*DeepDescMutex, *friend_robot->DeepDescMutex) == -1) {
                if(last_inter_lcd_size==deepDescManager.deepdesc_invkeys_mat_.size()){
                    friend_robot->DeepDescMutex->unlock();
                    DeepDescMutex->unlock();
                    continue;
                }

                auto detectResult = detectDeepLoopFriendRobot(*this, friend_robot);

                friend_robot->DeepDescMutex->unlock();
                DeepDescMutex->unlock();

                int curr_robot_id = std::get<0>(detectResult);
                int loop_id = std::get<2>(detectResult);
                float yaw_diff_rad = std::get<3>(detectResult);

                if (loop_id != -1) {
                    const int curr_node_idx = last_inter_lcd_size - 1;  // 当前节点索引
                    // if (curr_robot_id == 1){
                    //     std::string save_path = "/home/cyw/temp/deep_loop.txt";
                    //     std::ofstream outfile;
                    //     outfile.open(save_path, std::ios_base::app);
                    //     outfile << " curr_key_idx:" << curr_node_idx << "loop_id:" << loop_id << "\n";
                    //     outfile.close();
                    // }

                    
                    // if (curr_robot_id == 0){
                    //     std::string save_path_1 = "/home/cyw/temp/deep_loop_1.txt";
                    //     std::ofstream outfile;
                    //     outfile.open(save_path_1, std::ios_base::app);
                    //     outfile << " curr_key_idx:" << curr_node_idx << "loop_id:" << loop_id << "\n";
                    //     outfile.close();
                    // }

                    // const int curr_node_idx = this->deepDescManager.deepdesc_invkeys_mat_.size() - 1;  // 当前节点索引
                    // const int curr_node_idx = last_inter_lcd_size - 1;  // 当前节点索引
                    const int friend_node_idx = loop_id;
                    // if (curr_robot_id == 1){
                    //     std::cout << "curr_node_idx:" << curr_node_idx << std::endl;
                    // }

                    // std::cout << "Candidate valid! Loop detected between " << "Robot " << curr_robot_id << ": " << curr_node_idx << 
                    //         " and " << " Friend Robot " << friend_robot_id << ": " << friend_node_idx << std::endl;

                    mBuf->lock();
                    interRobotLoopICPBuf.push(std::make_tuple(curr_robot_id, friend_robot_id, curr_node_idx, friend_node_idx));
                    // allinterRobotLoopICPBuf.push(std::make_tuple(curr_robot_id, friend_robot_id, curr_node_idx, friend_node_idx));
                    mBuf->unlock();
                }
            }
        }
    }

    std::tuple<int, int, int, float> detectDeepLoopFriendRobot(Robot& robot, std::shared_ptr<Robot>& friend_robot) {
        int loop_id = -1;
        int friend_robot_idx = -1;
        float yaw_diff_rad = 0.0;
        int nn_idx = 0;
        const int MIN_DESC_COUNT = 10; // 设定最少描述子数量

        // std::cout << " friend_robot_id: " << friend_robot->robot_id << "tree_making_counter: " << friend_robot->tree_making_counter << std::endl;

        if (friend_robot->deepDescManager.deepdesc_invkeys_mat_.size() < MIN_DESC_COUNT)
        {
            ROS_WARN("Not enough descriptors for friend_robot_id: %d (only %ld descriptors)",
                     friend_robot->robot_id,
                     friend_robot->deepDescManager.deepdesc_invkeys_mat_.size());
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

            if (friend_robot->deepDescManager.deepdesc_invkeys_to_search_.empty()) {
                ROS_WARN("deepdesc_invkeys_to_search_ is empty. Skipping KD-tree construction for friend_robot_id: %d", friend_robot->robot_id);
                return std::make_tuple(robot.robot_id, friend_robot->robot_id, loop_id, yaw_diff_rad);  // **直接返回**
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

        if(robot.last_inter_lcd_size==robot.deepDescManager.deepdesc_invkeys_mat_.size()){
            return std::make_tuple(robot.robot_id, friend_robot_idx, loop_id, yaw_diff_rad);
        }

        // robot.last_inter_lcd_size = robot.deepDescManager.deepdesc_invkeys_mat_.size();
        // auto curr_key = robot.deepDescManager.deepdesc_invkeys_mat_.back();

        robot.last_inter_lcd_size += 1;
        auto curr_key = robot.deepDescManager.deepdesc_invkeys_mat_[robot.last_inter_lcd_size-1];

        // KD-tree 最近邻搜索
        std::vector<size_t> candidate_indexes(friend_robot->deepDescManager.NUM_CANDIDATES_FROM_TREE);
        std::vector<float> out_dists_sqr(friend_robot->deepDescManager.NUM_CANDIDATES_FROM_TREE);

        nanoflann::KNNResultSet<float> knnsearch_result(friend_robot->deepDescManager.NUM_CANDIDATES_FROM_TREE);
        knnsearch_result.init(&candidate_indexes[0], &out_dists_sqr[0]);

        friend_robot->deepDescManager.deepdesc_tree_->index->findNeighbors(knnsearch_result, &curr_key[0], nanoflann::SearchParams(10));

        // check code for debug

        // std::cout << "KNN distances and candidate IDs: ";
        // for (size_t i = 0; i < out_dists_sqr.size(); i++)
        // {
        //     if (out_dists_sqr[i] == 0.0)
        //     {
        //         std::cout << "[ID: " << candidate_indexes[i] << ", Dist: " << out_dists_sqr[i] << "] ";
        //     }
        // }
        // std::cout << std::endl;

        for (int candidate_iter_idx = 0; candidate_iter_idx < deepDescManager.NUM_CANDIDATES_FROM_TREE; candidate_iter_idx++) {
            int candidate_idx = deepDescManager.TREE_MAKE_STEP_ * candidate_indexes[candidate_iter_idx];
            float candidate_dist = out_dists_sqr[candidate_iter_idx];

            if (candidate_dist == 0.0 || candidate_idx < 0 || candidate_idx >= friend_robot->deepDescManager.deepdesc_invkeys_mat_.size()) {
                continue;
            }

            if (candidate_dist < min_dist) {
                min_dist = candidate_dist;
                friend_robot_idx = friend_robot->robot_id;
                nn_idx = candidate_idx;
            }
        }

        if (min_dist == 0.0)
        {
            // 打印当前帧描述子矩阵 size 和 友机描述子矩阵 size
            std::cout << "robot_id: " << robot.robot_id << " friend_robot_id: " << friend_robot->robot_id << std::endl;
            std::cout << "robot_desc_mat_size: " << robot.deepDescManager.deepdesc_invkeys_mat_.size() << std::endl;
            std::cout << "friend_robot_desc_mat_size: " << friend_robot->deepDescManager.deepdesc_invkeys_mat_.size() << std::endl;
        
            // 检查友机 KD-tree 结构是否有效
            std::cout << "friend_robot KD-tree size: " << friend_robot->deepDescManager.deepdesc_invkeys_to_search_.size() << std::endl;
            if (!friend_robot->deepDescManager.deepdesc_tree_) {
                std::cout << "[Warning] Friend robot KD-tree is NULL!" << std::endl;
            } else {
                std::cout << "[Info] Friend robot KD-tree exists." << std::endl;
            }
        }

        // std::cout << "robot_id:" << robot.robot_id << ", friend_robot_id:" << friend_robot->robot_id
        // << ", loop_id: " << loop_id << " ,min_dist: " << min_dist << std::endl;
        // std::string save_path = "/home/cyw/temp/deep_loop_0.txt";
        // if (robot.robot_id == 0){ 
        //     std::ofstream outfile;
        //     outfile.open(save_path, std::ios_base::app);
        //     outfile << "robot_id:" << robot.robot_id << " curr_key_idx:" << robot.deepDescManager.deepdesc_invkeys_mat_.size() - 1 << "\n";
        //     outfile.close();
        // }

        // 判断距离是否小于设定阈值
        if (min_dist < robot.deepDescManager.Multi_DeepLOOP_DIST_THRES) {
            loop_id = nn_idx;
            std::cout << "robot_id: " << robot.robot_id << " friend_robot_id: " << friend_robot->robot_id 
                  << " loop_id: " << loop_id << " min_dist: " << min_dist << std::endl;
            std::cout << "curr_node_idx: " << robot.last_inter_lcd_size - 1 << std::endl;

            // std::string save_path = "/home/cyw/temp/deep_loop.txt";
            // std::ofstream outfile;
            // outfile.open(save_path, std::ios_base::app);
            // outfile << "robot_id:" << robot.robot_id << ", friend_robot_id:" << friend_robot->robot_id
            //         << ", loop_id: " << loop_id << " ,min_dist: " << min_dist << "\n";
            // outfile.close();
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
        markerNode.scale.x = 0.3;
        markerNode.scale.y = 0.3;
        markerNode.scale.z = 0.3;
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
        markerEdge.scale.x = 0.5;
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
        markerNode.scale.x = 0.3;
        markerNode.scale.y = 0.3;
        markerNode.scale.z = 0.3;
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
        interMarkerEdge.scale.x = 0.5;
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
            }
            else if (lcd_mode == 2){
                perfromDeepDescLoopClosure();
            }

            visualizeLoopClosure();
            
            // if (*shutdown)
            //     break;
        }
    }

    void process_inter_lcd()
    {
        // float loopClosureFrequency = 1.0;
        ros::Rate rate(loopClosureFrequency);
        while (ros::ok())
        // while (1)
        {
            rate.sleep();
            // std::cout << "process_lcd still running" << std::endl;
            if (lcd_mode == 1){
                performFriendSCLoopClosure();
            }
            else if (lcd_mode == 2){
                performFriendDeepDescLoopClosure();
            }
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

                // 构建一个txt文件，记录每次的闭环的两帧,不同robot存储不同文件
                std::string loop_path = save_directory + std::to_string(robot_id) + "_loop.txt";
                std::string command = "mkdir -p " + save_directory;
                if (system(command.c_str()) != 0)
                {
                    ROS_ERROR_STREAM("Failed to create directory: " << save_directory);
                    continue;
                }

                if (relative_pose_optional)
                {
                    auto robot_id = this->robot_id;

                    gtsam::Pose3 relative_pose = relative_pose_optional.value();

                    // modify the relative pose x, y, z as 0
                    // gtsam::Point3 modified_translation(0, 0, 0);
                    // relative_pose = gtsam::Pose3(relative_pose.rotation(), modified_translation);

                    mtxPosegraph->lock();
                    const int curr_global_node_idx = index_offset + curr_node_idx;
                    const int prev_global_node_idx = index_offset + prev_node_idx;

                    gtSAMgraph->add(gtsam::BetweenFactor<gtsam::Pose3>(prev_global_node_idx, curr_global_node_idx, relative_pose, robustLoopNoise));
                    // gtSAMgraph->add(gtsam::BetweenFactor<gtsam::Pose3>(prev_global_node_idx, curr_global_node_idx, relative_pose, odomNoise));
                    writeEdge({prev_node_idx, curr_node_idx}, relative_pose);

                    // 写入闭环信息到文件
                    {
                        std::ofstream outfile(loop_path, std::ios_base::app);
                        if (outfile.is_open())
                        {
                            std::lock_guard<std::mutex> lock(*mKF); // 确保线程安全
                            outfile << "robot_id:" << robot_id
                                    << ", curr_global_node_idx:" << curr_global_node_idx
                                    << ", prev_global_node_idx:" << prev_global_node_idx << "\n";
                            outfile << "relative_pose:\n"
                                    << relative_pose.matrix() << "\n\n";
                            outfile.close();
                        }
                        else
                        {
                            ROS_ERROR_STREAM("Failed to open file: " << loop_path);
                        }
                    }

                    // mKF->lock();
                    // std::cout << "curr_global_node_idx:" << curr_global_node_idx << ", prev_global_node_idx: " << prev_global_node_idx << std::endl;
                    // scanMatchStream << keyframeTimes[prev_node_idx] << " " << keyframeTimes[curr_node_idx] << "\n";
                    // mKF->unlock();
                    mtxPosegraph->unlock();
                }
            }

            std::chrono::milliseconds dura(2);
            std::this_thread::sleep_for(dura);

            // if (*shutdown)
            //     break;
        }
    }

    // void process_inter_icp()
    // {
    //     while (ros::ok())
    //     {
    //         // std::cout << "process_inter_icp still running" << std::endl;
    //         while (!interRobotLoopICPBuf.empty()) {
    //             if (interRobotLoopICPBuf.size() > 30) {
    //                 ROS_WARN("Too many inter-robot loop clousre candidates to be ICPed is waiting, size: %d ... ", interRobotLoopICPBuf.size());
    //             }

    //             auto loop_idx_tuple = interRobotLoopICPBuf.front();
    //             mBuf->lock();
    //             interRobotLoopICPBuf.pop();
    //             mBuf->unlock();
    //             // std::cout << "after_interRobotLoopICPBuf.size(): " << interRobotLoopICPBuf.size() << std::endl;

    //             const int curr_robot_id = std::get<0>(loop_idx_tuple);
    //             const int friend_robot_id = std::get<1>(loop_idx_tuple);
    //             const int curr_node_idx = std::get<2>(loop_idx_tuple);
    //             const int friend_node_idx = std::get<3>(loop_idx_tuple);

    //             int curr_global_node_idx = curr_robot_id * MAX_NODES_PER_ROBOT + curr_node_idx;
    //             int friend_global_node_idx = friend_robot_id * MAX_NODES_PER_ROBOT + friend_node_idx;

    //             std::cout << "curr_global_node_idx:" << curr_global_node_idx << 
    //             ", friend_global_node_idx:" << friend_global_node_idx <<std::endl;
                
    //             Robot* curr_robot = this;
    //             Robot* friend_robot = curr_robot->friend_robots[friend_robot_id].get();

    //             if (curr_robot->robot_id == 1){
    //                 std::string save_path = "/home/cyw/temp/robot1_index.txt";
    //                 std::ofstream outfile;
    //                 outfile.open(save_path, std::ios_base::app);
    //                 outfile << "curr_global_node_idx:" << curr_global_node_idx << ", friend_global_node_idx:" << friend_global_node_idx << "\n";
    //                 outfile.close();
    //             }

    //             // bool high_candidate_region = isHighCandidateRegion(allinterRobotLoopICPBuf, curr_node_idx);
    //             // if (high_candidate_region) {
    //             //     std::cout << "High candidate region detected, adjusting Quatro frequency." << std::endl;
    //             //     useQuatro = true;
    //             // if (curr_robot->last_inter_lcd_size % 5 == 0) {
    //                 // std::cout << "normal frequency quatro" << std::endl;
    //             //     useQuatro = true;
    //             // } else {
    //             //     useQuatro = false;
    //             // }

    //             std::cout << "useQuatro:" << useQuatro<< std::endl;

    //             auto inter_relative_pose_optional = doICPVirtualRelativeFriendRobot(curr_robot, friend_robot, curr_node_idx, friend_node_idx);

    //             if (inter_relative_pose_optional)
    //             {
    //                 gtsam::Pose3 inter_relative_pose = inter_relative_pose_optional.value();

    //                 if ((curr_robot_id != 0 && friend_robot_id != 0) && *inter_init == false)
    //                 {
    //                     // 保存多机器人闭环约束到容器中
    //                     {
    //                         std::lock_guard<std::mutex> lock(*interloopPairMutex);
    //                         inter_loop_container.emplace_back(curr_global_node_idx, friend_global_node_idx, inter_relative_pose);
    //                     }
    //                 }
    //                 else
    //                 {
    //                     // 获取当前机器人和友机的关键帧位姿
    //                     Pose6D curr_pose6d = curr_robot->keyframePoses[curr_node_idx];
    //                     Pose6D friend_pose6d = friend_robot->keyframePoses[friend_node_idx];

    //                     gtsam::Pose3 curr_pose3 = convertPose6DToPose3(curr_pose6d);
    //                     gtsam::Pose3 friend_pose3 = convertPose6DToPose3(friend_pose6d);

    //                     gtsam::Pose3 inter_relative_pose_world = friend_pose3 * inter_relative_pose * curr_pose3.inverse();

    //                     mtxPosegraph->lock();
    //                     gtSAMgraph->add(gtsam::BetweenFactor<gtsam::Pose3>(
    //                         friend_global_node_idx, curr_global_node_idx, inter_relative_pose, robustLoopNoise));
    //                     mtxPosegraph->unlock();

    //                     // 保存闭环信息到文件
    //                     {
    //                         std::lock_guard<std::mutex> lock(*mKF);
    //                         robots_scanMatchStream << curr_robot_id << " " << keyframeTimes[curr_node_idx] << " "
    //                                                << friend_robot_id << " " << keyframeTimes[friend_node_idx] << "\n";
    //                     }

    //                     // 如果是第一次初始化，处理所有闭环约束
    //                     if (*inter_init == false)
    //                     {
    //                         std::lock_guard<std::mutex> lock(*interloopPairMutex);
    //                         for (const auto &entry : inter_loop_container)
    //                         {
    //                             int curr_global_node_idx = std::get<0>(entry);
    //                             int friend_global_node_idx = std::get<1>(entry);
    //                             gtsam::Pose3 inter_relative_pose = std::get<2>(entry);

    //                             gtSAMgraph->add(gtsam::BetweenFactor<gtsam::Pose3>(
    //                                 curr_global_node_idx, friend_global_node_idx, inter_relative_pose, robustLoopNoise));
    //                         }
    //                         *inter_init = true;
    //                     }
    //                 }
    //             }
    //         }

    //         std::chrono::milliseconds dura(2);
    //         std::this_thread::sleep_for(dura);

    //         // if (*shutdown)
    //         //     break;
    //     }
    // }

    void process_inter_icp()
    {
        const int batch_size = 10; // 每次处理的任务数量

        while (ros::ok())
        {
            std::vector<std::tuple<int, int, int, int>> tasks;

            // 从队列中取出固定数量的任务
            {
                if (interRobotLoopICPBuf.size() > 30)
                {
                    ROS_WARN("Too many inter-robot loop clousre candidates to be ICPed is waiting, size: %d ... ", interRobotLoopICPBuf.size());
                }
                std::lock_guard<std::mutex> lock(*mBuf);
                while (!interRobotLoopICPBuf.empty() && tasks.size() < batch_size)
                {
                    tasks.push_back(interRobotLoopICPBuf.front());
                    interRobotLoopICPBuf.pop();
                }
            }

            // 遍历任务并逐个处理
            for (const auto &loop_idx_tuple : tasks)
            {
                const int curr_robot_id = std::get<0>(loop_idx_tuple);
                const int friend_robot_id = std::get<1>(loop_idx_tuple);
                const int curr_node_idx = std::get<2>(loop_idx_tuple);
                const int friend_node_idx = std::get<3>(loop_idx_tuple);

                int curr_global_node_idx = curr_robot_id * MAX_NODES_PER_ROBOT + curr_node_idx;
                int friend_global_node_idx = friend_robot_id * MAX_NODES_PER_ROBOT + friend_node_idx;

                std::cout << "curr_global_node_idx:" << curr_global_node_idx
                          << ", friend_global_node_idx:" << friend_global_node_idx << std::endl;

                Robot *curr_robot = this;
                Robot *friend_robot = curr_robot->friend_robots[friend_robot_id].get();

                // if (curr_robot->robot_id == 1)
                // {
                //     std::string save_path = "/home/cyw/temp/robot1_index.txt";
                //     std::ofstream outfile;
                //     outfile.open(save_path, std::ios_base::app);
                //     outfile << "curr_global_node_idx:" << curr_global_node_idx << ", friend_global_node_idx:" << friend_global_node_idx << "\n";
                //     outfile.close();
                // }

                std::cout << "useQuatro:" << useQuatro << std::endl;

                auto inter_relative_pose_optional = doICPVirtualRelativeFriendRobot(curr_robot, friend_robot, curr_node_idx, friend_node_idx);

                if (inter_relative_pose_optional)
                {
                    gtsam::Pose3 inter_relative_pose = inter_relative_pose_optional.value();

                    if ((curr_robot_id != 0 && friend_robot_id != 0) && *inter_init == false)
                    {
                        // 保存多机器人闭环约束到容器中
                        {
                            std::lock_guard<std::mutex> lock(*interloopPairMutex);
                            inter_loop_container.emplace_back(curr_global_node_idx, friend_global_node_idx, inter_relative_pose);
                        }
                    }
                    else
                    {
                        // 获取当前机器人和友机的关键帧位姿
                        Pose6D curr_pose6d = curr_robot->keyframePoses[curr_node_idx];
                        Pose6D friend_pose6d = friend_robot->keyframePoses[friend_node_idx];

                        gtsam::Pose3 curr_pose3 = convertPose6DToPose3(curr_pose6d);
                        gtsam::Pose3 friend_pose3 = convertPose6DToPose3(friend_pose6d);

                        gtsam::Pose3 inter_relative_pose_world = friend_pose3 * inter_relative_pose * curr_pose3.inverse();

                        mtxPosegraph->lock();
                        gtSAMgraph->add(gtsam::BetweenFactor<gtsam::Pose3>(
                            friend_global_node_idx, curr_global_node_idx, inter_relative_pose, robustLoopNoise));
                        mtxPosegraph->unlock();

                        // 保存闭环信息到文件
                        {
                            std::lock_guard<std::mutex> lock(*mKF);
                            robots_scanMatchStream << curr_robot_id << " " << keyframeTimes[curr_node_idx] << " "
                                                   << friend_robot_id << " " << keyframeTimes[friend_node_idx] << "\n";
                        }

                        // 如果是第一次初始化，处理所有闭环约束
                        if (*inter_init == false)
                        {
                            std::lock_guard<std::mutex> lock(*interloopPairMutex);
                            for (const auto &entry : inter_loop_container)
                            {
                                int curr_global_node_idx = std::get<0>(entry);
                                int friend_global_node_idx = std::get<1>(entry);
                                gtsam::Pose3 inter_relative_pose = std::get<2>(entry);

                                gtSAMgraph->add(gtsam::BetweenFactor<gtsam::Pose3>(
                                    curr_global_node_idx, friend_global_node_idx, inter_relative_pose, robustLoopNoise));
                            }
                            *inter_init = true;
                        }
                    }
                }
            }

            // 休眠一段时间，避免占用过多 CPU 资源
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
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
                    // std::cout << "Saving the posegraph for" << robot_id << " ..." << std::endl;
                    // ROS_INFO_STREAM("Saving odometry poses to file " << odomTUMFormat); 
                    saveOdometryVerticesTUMFormat(odomTUMFormat, keyframePoses, keyframeTimes, robot_id);
                    
                    // ROS_INFO_STREAM("Saving optimized poses to file " << pgTUMFormat);
                    saveOptimizedVerticesTUMFormat(isamCurrentEstimate, pgTUMFormat, isamStateTimes, robot_id);
                    // break;
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

    bool *inter_init_global = new bool(false);
    std::vector<std::tuple<int, int, gtsam::Pose3>> *inter_loop_container_global = new std::vector<std::tuple<int, int, gtsam::Pose3>>();
    
    int num_robots = 3; // 假设有三个机器人

    std::vector<std::shared_ptr<Robot>> robots;
    for (int i = 0; i < num_robots; ++i)
    {
        robots.emplace_back(std::make_shared<Robot>(nh, i, isam_global, mtxPosegraph_global, inter_init_global, *inter_loop_container_global));
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
        processing_threads.emplace_back(&Robot::process_inter_lcd, robot); // inter-robot loop closure detect
        processing_threads.emplace_back(&Robot::process_icp, robot); // icp pose get (for lc)
        processing_threads.emplace_back(&Robot::process_inter_icp, robot); // icp pose get (for inter lc)
        processing_threads.emplace_back(&Robot::process_isam, robot);
        processing_threads.emplace_back(&Robot::process_viz_path, robot);
        processing_threads.emplace_back(&Robot::process_viz_map, robot);

        // processing_threads.emplace_back(&Robot::process_dense_pgo, robot); // process_dense_pgo
        // processing_threads.emplace_back(&Robot::add_reloc_desc, robot);
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
