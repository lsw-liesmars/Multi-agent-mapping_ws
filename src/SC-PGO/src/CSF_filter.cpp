#include "CSF_filter.h"
#include <pcl/filters/filter.h>
#include <pcl/visualization/cloud_viewer.h>

void applyCSFFilter(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_in,
                    pcl::PointCloud<pcl::PointXYZI>& ground_out,
                    pcl::PointCloud<pcl::PointXYZI>::Ptr& non_ground_out,
                    const CSFParams& params) {

    CSF csf;
    
    // 转换PCL点云到CSF格式
    std::vector<csf::Point> csf_points;
    csf_points.reserve(cloud_in->size());
    for (const auto& pt : cloud_in->points) {
        csf_points.emplace_back(pt.x, pt.y, pt.z);
    }
    csf.setPointCloud(csf_points);

    // std::cout << "CSF Parameters set. " << std::endl;
    csf.params.bSloopSmooth = params.bSloopSmooth;
    csf.params.cloth_resolution = params.cloth_resolution;
    csf.params.rigidness = params.rigidness;
    csf.params.time_step = params.time_step;
    csf.params.class_threshold = params.class_threshold;
    csf.params.interations = params.iterations;

    std::vector<int> groundIndexes, offGroundIndexes;
    csf.do_filtering(groundIndexes, offGroundIndexes);

    pcl::copyPointCloud(*cloud_in, groundIndexes, ground_out);
    pcl::copyPointCloud(*cloud_in, offGroundIndexes, *non_ground_out);
    // std::cout << "CSF Filtering done. " << std::endl;
}
