#include<Eigen/Core>
#include <Eigen/Geometry>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <mutex>
#include <omp.h>
#include <pcl/point_cloud.h>

// TODO yiwen: copy function used in MultiRobotlaserPosegraphOptimization.cpp form laserPosegraphOptimization copy.cpp;
// then bianyi sc-pgo. 

#define MAX_NODES_PER_ROBOT 10000

using namespace gtsam;

std::mutex mKF;

std::vector<std::string> edges_str;
std::vector<std::string> vertices_str;

// int historyKeyframeSearchNum;
// double loopFitnessScoreThreshold;  // user parameter but fixed low value is safe, e.g., 0.3

std::string padZeros(int val, int num_digits = 6) {
  std::ostringstream out;
  out << std::internal << std::setfill('0') << std::setw(num_digits) << val;
  return out.str();
}

float diffRotation(const Eigen::Matrix3f &a, const Eigen::Matrix3f &b)
{
    Eigen::Matrix3f c = a.transpose() * b;
    float trace = c(0,0) + c(1,1) + c(2,2);
    float angle = acos((trace - 1.0) / 2.0);
    return angle;
}

Eigen::Affine3f toEigenAffine3f(const Pose6D &pose) {
    return pcl::getTransformation(pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw);
}

pcl::PointCloud<PointType>::Ptr local2global(const pcl::PointCloud<PointType>::Ptr &cloudIn, const Pose6D& tf,
                                            const Eigen::Vector3d &transPos = Eigen::Vector3d(0,0,0))
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(tf.x, tf.y, tf.z, tf.roll, tf.pitch, tf.yaw);
    
    int numberOfCores = 16;
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3) - transPos(0);
        cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3) - transPos(1);
        cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3) - transPos(2);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }

    return cloudOut;
}

Pose6D getOdom(nav_msgs::Odometry::ConstPtr _odom)
{
    auto tx = _odom->pose.pose.position.x;
    auto ty = _odom->pose.pose.position.y;
    auto tz = _odom->pose.pose.position.z;

    double roll, pitch, yaw;
    geometry_msgs::Quaternion quat = _odom->pose.pose.orientation;
    tf::Matrix3x3(tf::Quaternion(quat.x, quat.y, quat.z, quat.w)).getRPY(roll, pitch, yaw);

    return Pose6D{tx, ty, tz, roll, pitch, yaw}; 
} // getOdom

Pose6D diffTransformation(const Pose6D& _p1, const Pose6D& _p2)
{
    Eigen::Affine3f SE3_p1 = pcl::getTransformation(_p1.x, _p1.y, _p1.z, _p1.roll, _p1.pitch, _p1.yaw);
    Eigen::Affine3f SE3_p2 = pcl::getTransformation(_p2.x, _p2.y, _p2.z, _p2.roll, _p2.pitch, _p2.yaw);
    Eigen::Matrix4f SE3_delta0 = SE3_p1.matrix().inverse() * SE3_p2.matrix();
    Eigen::Affine3f SE3_delta; SE3_delta.matrix() = SE3_delta0;
    float dx, dy, dz, droll, dpitch, dyaw;
    pcl::getTranslationAndEulerAngles (SE3_delta, dx, dy, dz, droll, dpitch, dyaw);
    // std::cout << "delta : " << dx << ", " << dy << ", " << dz << ", " << droll << ", " << dpitch << ", " << dyaw << std::endl;

    return Pose6D{double(abs(dx)), double(abs(dy)), double(abs(dz)), double(abs(droll)), double(abs(dpitch)), double(abs(dyaw))};
} // SE3Diff

gtsam::Pose3 Pose6DtoGTSAMPose3(const Pose6D& p)
{
    return gtsam::Pose3( gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw), gtsam::Point3(p.x, p.y, p.z) );
} // Pose6DtoGTSAMPose3

gtsam::Pose3 Pose6dTogtsamPose3(Pose6D pose)
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(pose.roll), double(pose.pitch), double(pose.yaw)),
                                gtsam::Point3(double(pose.x),    double(pose.y),     double(pose.z)));
}

void writeVertex(const int _node_idx, const gtsam::Pose3& _initPose)
{
    gtsam::Point3 t = _initPose.translation();
    gtsam::Rot3 R = _initPose.rotation();

    std::string curVertexInfo {
        "VERTEX_SE3:QUAT " + std::to_string(_node_idx) + " "
        + std::to_string(t.x()) + " " + std::to_string(t.y()) + " " + std::to_string(t.z())  + " " 
        + std::to_string(R.toQuaternion().x()) + " " + std::to_string(R.toQuaternion().y()) + " " 
        + std::to_string(R.toQuaternion().z()) + " " + std::to_string(R.toQuaternion().w()) };

    // pgVertexSaveStream << curVertexInfo << std::endl;
    vertices_str.emplace_back(curVertexInfo);
}

void writeEdge(const std::pair<int, int> _node_idx_pair, const gtsam::Pose3& _relPose)
{
    gtsam::Point3 t = _relPose.translation();
    gtsam::Rot3 R = _relPose.rotation();

    std::string curEdgeInfo {
        "EDGE_SE3:QUAT " + std::to_string(_node_idx_pair.first) + " " + std::to_string(_node_idx_pair.second) + " "
        + std::to_string(t.x()) + " " + std::to_string(t.y()) + " " + std::to_string(t.z())  + " " 
        + std::to_string(R.toQuaternion().x()) + " " + std::to_string(R.toQuaternion().y()) + " " 
        + std::to_string(R.toQuaternion().z()) + " " + std::to_string(R.toQuaternion().w()) };

    // pgEdgeSaveStream << curEdgeInfo << std::endl;
    edges_str.emplace_back(curEdgeInfo);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr vector2pc(const std::vector<Pose6D> vectorPose6d){
    pcl::PointCloud<pcl::PointXYZ>::Ptr res( new pcl::PointCloud<pcl::PointXYZ> ) ;
    for( auto p : vectorPose6d){
        res->points.emplace_back(p.x, p.y, p.z);
    }
    return res;
}

void saveOptimizedVerticesTUMFormat(const gtsam::Values &_estimates, std::string _filename, const std::vector<ros::Time> &isamStateTimes, int robot_id)
{
    // mKF.lock();
    // using namespace gtsam;    if (_estimates.size() != isamStateTimes.size()) {
    //     ROS_ERROR_STREAM("Inconsistent keyframe estimated #poses " << _estimates.size() 
    //             << " and #times " << isamStateTimes.size() << ".");
    // }
    std::fstream stream(_filename.c_str(), std::fstream::out);
    stream << std::fixed; // 设置固定格式
    size_t minsize = std::min(_estimates.size(), isamStateTimes.size());
    size_t i = 0;
    for (const auto& key_value : _estimates) {
        auto p = dynamic_cast<const GenericValue<Pose3>*>(&key_value.value);
        if (!p) continue;
        if (i >= minsize) break;
        const Pose3& pose = p->value();

        Point3 t = pose.translation();
        Eigen::Quaterniond q = pose.rotation().toQuaternion();
        int global_node_id = key_value.key;
        // std::cout << "global_node_id: " << global_node_id << std::endl;
        int local_node_id = global_node_id % MAX_NODES_PER_ROBOT;
        int robot_id_from_key = global_node_id / MAX_NODES_PER_ROBOT;
        // std::cout << "robot_id_from_key: " << robot_id_from_key << " robot_id: " << robot_id << std::endl;
        if (robot_id_from_key == robot_id) {
            stream << isamStateTimes[local_node_id].toSec() << " " << std::fixed << std::setprecision(8) << t.x() << " " << t.y() << " " << t.z()
                   << " " << std::setprecision(9) << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
            ++i;
        }
    }
    // mKF.unlock();
    stream.close();
}

void saveOdometryVerticesTUMFormat(std::string _filename, const std::vector<Pose6D> &keyframePoses, const std::vector<ros::Time> &keyframeTimes, int robot_id)
{
    // mKF.lock();
    if (keyframePoses.size() != keyframeTimes.size()) {
        ROS_ERROR_STREAM("Inconsistent keyframe #poses " << keyframePoses.size() 
                << " and #times " << keyframeTimes.size() << ".");
    }
    size_t minsize = std::min(keyframePoses.size(), keyframeTimes.size());
    std::fstream stream(_filename.c_str(), std::fstream::out);
    for (size_t i = 0; i < minsize; ++i) {
        const auto& _pose6d = keyframePoses[i];
        gtsam::Pose3 pose = Pose6DtoGTSAMPose3(_pose6d);
        Point3 t = pose.translation();
        Eigen::Quaterniond q = pose.rotation().toQuaternion();
        int global_node_id = robot_id * MAX_NODES_PER_ROBOT + i;
        stream << keyframeTimes[i].toNSec() << " " << std::fixed << std::setprecision(8) << t.x() << " " << t.y() << " " << t.z()
               << " " << std::setprecision(9) << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    }
    // mKF.unlock();
    stream.close();
}


pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, Eigen::Affine3f transCur)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    PointType *pointFrom;

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    int numberOfCores = 8; // TODO move to yaml 
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        pointFrom = &cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom->x + transCur(0,1) * pointFrom->y + transCur(0,2) * pointFrom->z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom->x + transCur(1,1) * pointFrom->y + transCur(1,2) * pointFrom->z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom->x + transCur(2,1) * pointFrom->y + transCur(2,2) * pointFrom->z + transCur(2,3);
        cloudOut->points[i].intensity = pointFrom->intensity;
    }
    return cloudOut;
}

pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, gtsam::Pose3 transformIn)
{
    Eigen::Affine3f transCur = pcl::getTransformation(
                                transformIn.translation().x(), transformIn.translation().y(), transformIn.translation().z(), 
                                transformIn.rotation().roll(), transformIn.rotation().pitch(), transformIn.rotation().yaw() );
    return transformPointCloud(cloudIn, transCur);
}

void saveSCD(std::string fileName, Eigen::MatrixXd matrix, std::string delimiter = " ")
{
    // delimiter: ", " or " " etc.

    int precision = 3; // or Eigen::FullPrecision, but SCD does not require such accruate precisions so 3 is enough.
    const static Eigen::IOFormat the_format(precision, Eigen::DontAlignCols, delimiter, "\n");
 
    std::ofstream file(fileName);
    if (file.is_open())
    {
        file << matrix.format(the_format);
        file.close();
    }
}