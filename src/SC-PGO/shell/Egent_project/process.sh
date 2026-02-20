#!/bin/bash
data_name="xinghu_2"
raw_data_dir="/media/cyw/Li_2T/China_Egypt_test/$data_name/" 
base_data_dir="/home/cyw/Chain_Egent_project/"
multi_data_dir="${base_data_dir}/${data_name}/"
#### 1.process single-agent odom  #####
echo "Processing single-agent odom"
/home/cyw/CYW/mapping/FAST_LIO_ws/src/FAST_LIO/shell/Egent/process_multi_mid360.sh "$raw_data_dir" "$multi_data_dir"

sleep 3

# check if there are three bags in the data_dir
if [ -d "$multi_data_dir" ]; then
    echo "Data directory exists: $multi_data_dir"
    
    # Count .bag files in the directory
    bag_count=$(find "$multi_data_dir" -name "*.bag" -type f | wc -l)
    
    if [ "$bag_count" -eq 3 ]; then
        echo "Found exactly 3 bag files in $multi_data_dir"
    else
        echo "Error: Expected 3 bag files, but found $bag_count in $multi_data_dir"
        exit 1
    fi
else
    echo "Data directory does not exist: $multi_data_dir"
    exit 1
fi

# 2. process multi-agent data
echo "Processing multi-agent mapping"

/home/cyw/CYW/mapping/Multi-agent-mapping_ws_yw/src/FAST_LIO_SLAM/SC-PGO/shell/Egent_project/process_mult_robots.sh "$multi_data_dir"


## move results to the target directory
python3 /home/cyw/CYW/mapping/Multi-agent-mapping_ws_yw/src/FAST_LIO_SLAM/SC-PGO/shell/Egent_project/move_results.py \
        --source /home/cyw/Desktop/temp/multi_agent_mapping_egypt \
        --dest "$base_data_dir/results/$data_name" 
sleep 2

# 4. # process PGO optimazion
# extract odom pose
/home/cyw/CYW/mapping/SLAM_tools/pointcloud_process/extract_all_poses.sh "$multi_data_dir"

sleep 2

# output final poses
/home/cyw/CYW/mapping/SLAM_tools/multi_pgo_agent/run_multi_robot_pgo.sh "$data_name" "$base_data_dir"
