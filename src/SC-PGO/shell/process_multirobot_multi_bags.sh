#!/bin/bash

export LD_LIBRARY_PATH=~/Documents/slam_devel/gtsam/install/lib:$LD_LIBRARY_PATH
source /home/cyw/CYW/mapping/Multi-agent-mapping_ws_yw/devel/setup.bash

data_root="/media/cyw/ZX2_CYW/mapping_data_formal/multi_agent_data"
group_names=("group6")  #"group5" "group6" "group7"
robot_num=3

bag_groups=()
for group in "${group_names[@]}"; do
    bags=""
    for ((r=0; r<robot_num; r++)); do
        bag_path="${data_root}/${group}/robot${r}.bag"
        if [ $r -eq 0 ]; then
            bags="$bag_path"
        else
            bags="${bags};${bag_path}"
        fi
    done
    bag_groups+=("$bags")
done

# 打印 bag_groups 内容
echo "bag_groups content:"
for ((i=0; i<${#bag_groups[@]}; i++)); do
    echo "  ${group_names[i]}: ${bag_groups[i]}"
done

output_base=~/Desktop/temp/multi_agent_mapping

for ((i=0; i<${#bag_groups[@]}; i++)); do
    output_dir=${output_base}/${group_names[i]}
    mkdir -p "$output_dir"

    rm -rf ${output_base}/successed_res.txt
    rm -rf ${output_dir}/failed_quatro.txt

    roslaunch aloam_velodyne aloam_mid360_multirobot_whu.launch save_directory:=$output_dir &
    LAUNCH_PID=$!
    sleep 2

    echo "Processing group ${group_names[i]} with bags: ${bag_groups[i]}"

    IFS=";" read -ra bags <<< "${bag_groups[i]}"

    echo "1111 ${group_names[i]} with bags: ${bags[0]}"
    rosbag play "${bags[0]}" -r 1.0 -s 0 -q &
    rosbag play "${bags[1]}" -r 1.0 -s 0 -q &
    rosbag play "${bags[2]}" -r 1.0 -s 0 -q &

    wait
    sleep 2
done
