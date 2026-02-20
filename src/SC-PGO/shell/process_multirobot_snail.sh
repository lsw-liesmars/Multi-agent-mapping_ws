#!/bin/bash
# ##
export LD_LIBRARY_PATH=~/Documents/slam_devel/gtsam/install/lib:$LD_LIBRARY_PATH
source /home/cyw/CYW/mapping/Multi-agent-mapping_ws_yw/devel/setup.bash

data_base="/media/cyw/ZX2_CYW/mapping_data_formal/multi_agent_data"
output_base=~/Desktop/temp/multi_agent_mapping_snail

output_dir="${output_base}/snail/robot2"
mkdir -p "$output_dir"

group_dir="${data_base}/snail"

rm -rf "${output_base}/successed_res.txt"
rm -rf "${output_dir}/failed_quatro.txt"
rm -rf "${output_dir}/successed_trans.txt"

# roslaunch aloam_velodyne aloam_mid360_multirobot_whu.launch &
roslaunch aloam_velodyne aloam_hesai32_multirobot_SUV.launch &
sleep 2
# small scenes
rosbag play "${group_dir}/data1_with_radar.bag" -r 1.0 -s 0 -q &
rosbag play "${group_dir}/data2_with_radar.bag" -r 1.0 -s 0 -q &
rosbag play "${group_dir}/data3_with_radar.bag" -r 1.0 -s 0 -q &

# roofdata
# rosbag play /media/cyw/ZX2_CYW/mapping_data/roofdata/data1/data1.bag -r 1.0 -s 0 -q &
# rosbag play /media/cyw/ZX2_CYW/mapping_data/roofdata/data2/data2.bag -r 1.0 -s 0 -q &
# rosbag play /media/cyw/ZX2_CYW/mapping_data/roofdata/data3/data3.bag -r 1.0 -s 0 -q &


wait

# catkin build -DCeres_DIR=/home/cyw/Documents/slam_devel/lib/cmake/Ceres -DGTSAM_DIR=/home/cyw/Documents/slam_devel/lib/cmake/GTSAM