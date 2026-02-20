# source /home/cyw/CYW/mapping/Muti-agent-mapping_ws/devel/setup.bash
# roslaunch fast_lio mapping_hesai32_whu_handheld.launch &
# sleep 2

rm -rf /home/cyw/temp/successed_res.txt
rm -rf /home/cyw/temp/failed_quatro.txt


# # ##
export LD_LIBRARY_PATH=~/Documents/slam_devel/gtsam/install/lib:$LD_LIBRARY_PATH
source /home/cyw/CYW/mapping/Multi-agent-mapping_ws_yw/devel/setup.bash
roslaunch aloam_velodyne aloam_mid360_multirobot.launch &
sleep 2


# # small scenes
rosbag play /media/cyw/data/data/20250410_odom/data1.bag -r 1.0 -s 0 -q &
rosbag play /media/cyw/data/data/20250410_odom/data2.bag -r 1.0 -s 0 -q &
rosbag play /media/cyw/data/data/20250410_odom/data3.bag -r 1.0 -s 0 -q &

wait

# catkin build -DCeres_DIR=/home/cyw/Documents/slam_devel/lib/cmake/Ceres -DGTSAM_DIR=/home/cyw/Documents/slam_devel/lib/cmake/GTSAM