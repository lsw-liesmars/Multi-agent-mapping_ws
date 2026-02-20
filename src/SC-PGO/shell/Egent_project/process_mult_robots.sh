#!/bin/bash

export LD_LIBRARY_PATH=~/Documents/slam_devel/gtsam/install/lib:$LD_LIBRARY_PATH
source /home/cyw/CYW/mapping/Multi-agent-mapping_ws_yw/devel/setup.bash

# 接受命令行参数，如果没有提供则使用默认值
data_root="${1:-/media/cyw/ZX2_CYW/mapping_data_formal/multi_agent_data}"
#
# data_root="/home/cyw/multi_agent_data"
robot_num=3

# 构建bag路径
bags=""
for ((r=0; r<robot_num; r++)); do
    bag_path="${data_root}/robot${r}.bag"
    if [ $r -eq 0 ]; then
        bags="$bag_path"
    else
        bags="${bags};${bag_path}"
    fi
done

# 打印bag路径
echo "Processing ${group_name} with bags: ${bags}"

output_base=~/Desktop/temp/multi_agent_mapping_egypt
output_dir=${output_base}
mkdir -p "$output_dir"

rm -rf ${output_base}/successed_res.txt
rm -rf ${output_dir}/failed_quatro.txt

roslaunch aloam_velodyne aloam_mid360_multirobot_whu_egypt.launch save_directory:=$output_dir &
LAUNCH_PID=$!
sleep 2


IFS=";" read -ra bag_array <<< "${bags}"

echo "Playing bags for ${group_name}: ${bag_array[0]}, ${bag_array[1]}, ${bag_array[2]}"
rosbag play "${bag_array[0]}" -r 1.0 -s 0 -q &
rosbag play "${bag_array[1]}" -r 1.0 -s 0 -q &
rosbag play "${bag_array[2]}" -r 1.0 -s 0 -q &

wait
sleep 2
