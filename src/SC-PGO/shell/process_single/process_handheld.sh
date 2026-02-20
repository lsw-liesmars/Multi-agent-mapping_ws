seq=$1
sudo rm -rf "/media/cyw/ZX2_CYW/mapping_data/Whu_handheld/data$seq/data$seq.bag"

source /home/cyw/CYW/mapping/Multi-agent-mapping_ws_yw/devel/setup.bash
roslaunch fast_lio mapping_hesai32_whu_handheld.launch &
sleep 2

out_pctopic="/robot_0/full_cloud"
out_odomtopic="/robot_0/odom"
# source activate pt
python /home/cyw/CYW/mapping/Multi-agent-mapping_ws_yw/src/FAST_LIO_SLAM/script/record_odom_and_pcd_realtime.py \
        /media/cyw/ZX2_CYW/mapping_data/Whu_handheld/data$seq/data$seq.bag \
        --topics /cloud_registered_body:PointCloud2 /Odometry:Odometry \
        --output_topics /cloud_registered_body:$out_pctopic /Odometry:$out_odomtopic &
sleep 2

rosbag play /media/cyw/ZX2_CYW/mapping_data/bag_split/20231109/data4_aligned/segment_1.bag -r 1 -s 0 -q &
# rosbag play /media/cyw/KESU/datasets/inhouse/handheld/20231007/data$seq*.bag -r 1 -s 0 -q &

wait

# seq=$1
# sudo rm -rf /media/cyw/KESU/mapping_data/roofdata/data$seq/data$seq.bag

# if [ "$seq" -eq 1 ]; then
#     source /home/cyw/CYW/mapping/Multi-agent-mapping_ws_yw/devel/setup.bash
#     roslaunch fast_lio mapping_multi_rooftop.launch &
#     sleep 2
# else
#     source /home/cyw/CYW/mapping/Multi-agent-mapping_ws_yw/devel/setup.bash
#     roslaunch fast_lio mapping_multi_rooftop_hesai.launch &
#     sleep 2
# fi

# # out_pctopic="/robot_$seq/full_cloud"
# # out_odomtopic="/robot_$seq/odom"
# out_pctopic="/robot_2/full_cloud"
# out_odomtopic="/robot_2/odom"
# source activate pt
# # python --version
# python /home/cyw/CYW/mapping/Multi-agent-mapping_ws_yw/src/FAST_LIO_SLAM/script/record_odom_and_pcd_realtime.py \
#         /media/cyw/ZX2_CYW/mapping_data/roofdata/data$seq/data$seq.bag \
#         --topics /cloud_registered_body:PointCloud2 /Odometry:Odometry \
#         --output_topics /cloud_registered_body:$out_pctopic /Odometry:$out_odomtopic &
# sleep 2

# rosbag play  /media/cyw/KESU/mapping_data/multi_agent_data/$seq/*.bag -r 2 -s 0 -q &

# # rosbag play  /media/cyw/KESU/mapping_data/multi_agent_data/1/IMU_AND_LIDAR_DATA_0_1.bag -r 2 -s 0 -q &
# # rosbag play /media/cyw/KESU/mapping_data/multi_agent_data/3/2024-06-14-23-46-24.bag -r 2 -s 0 -q &
# # rosbag play /media/cyw/KESU/mapping_data/multi_agent_data/2/IMU_AND_LIDAR_DATA_1_1.bag -r 2 -s 0 -q &

# wait