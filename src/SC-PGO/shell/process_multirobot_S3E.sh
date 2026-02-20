# source /home/cyw/CYW/mapping/Muti-agent-mapping_ws/devel/setup.bash
# roslaunch fast_lio mapping_hesai32_whu_handheld.launch &
# sleep 2

rm -rf /home/cyw/temp/successed_res.txt
rm -rf /home/cyw/temp/failed_quatro.txt

# # ##
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
source /home/lsw/Multi-agent-mapping_ws/devel/setup.bash

# # Start roscore in background if not already running
# roscore &
# sleep 3

roslaunch aloam_velodyne aloam_vlp16_S3E.launch &

sleep 2

rosbag play /home/lsw/lvshangwei/cyw_handover/S3E_Playground_2/data1.bag -r 1.0 -s 0 -q &
rosbag play /home/lsw/lvshangwei/cyw_handover/S3E_Playground_2/data2.bag -r 1.0 -s 0 -q &
rosbag play /home/lsw/lvshangwei/cyw_handover/S3E_Playground_2/data3.bag -r 1.0 -s 0 -q &

wait
