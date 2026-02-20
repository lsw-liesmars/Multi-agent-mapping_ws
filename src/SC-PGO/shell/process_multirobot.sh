#!/bin/bash
# ##
export LD_LIBRARY_PATH=~/Documents/slam_devel/gtsam/install/lib:$LD_LIBRARY_PATH
source /home/cyw/CYW/mapping/Multi-agent-mapping_ws_yw/devel/setup.bash

data_base="/media/cyw/ZX2_CYW/mapping_data_formal/multi_agent_data"
output_base=~/Desktop/temp/multi_agent_mapping_temp
group_num=7
output_dir="${output_base}/group${group_num}/robot2"
mkdir -p "$output_dir"

group_dir="${data_base}/group${group_num}"

rm -rf "${output_base}/successed_res.txt"
rm -rf "${output_dir}/failed_quatro.txt"
rm -rf "${output_dir}/successed_trans.txt"

roslaunch aloam_velodyne aloam_mid360_multirobot_whu.launch &

sleep 2
# small scenes
rosbag play "${group_dir}/robot0.bag" -r 1.0 -s 0 -q &
rosbag play "${group_dir}/robot1.bag" -r 1.0 -s 0 -q &
rosbag play "${group_dir}/robot2.bag" -r 1.0 -s 0 -q &


wait
