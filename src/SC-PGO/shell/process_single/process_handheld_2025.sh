#!/bin/bash

# 输入三个bag文件路径
bag1=$1
bag2=$2
bag3=$3

# 要处理的bag文件路径数组
bags=($bag1 $bag2 $bag3)

# roslaunch初始化
source /home/cyw/CYW/mapping/Multi-agent-mapping_ws_yw/devel/setup.bash
roslaunch fast_lio mapping_hesai32_whu_handheld.launch &
sleep 2

# 循环处理每个bag文件
for i in {0..2}
do
    # 设置输出话题，根据机器人编号（i）
    out_pctopic="/robot_$(($i))"/full_cloud
    out_odomtopic="/robot_$(($i))"/odom

    # 删除已经存在的bag文件 (确保之前的文件已清理)
    sudo rm -rf "/media/cyw/ZX2_CYW/mapping_data/Whu_handheld/data$(($i+1))/data$(($i+1)).bag"
    
    # 启动Python脚本并处理当前的bag
    python /home/cyw/CYW/mapping/Multi-agent-mapping_ws_yw/src/FAST_LIO_SLAM/script/record_odom_and_pcd_realtime.py \
            "/media/cyw/ZX2_CYW/mapping_data/Whu_handheld/data$(($i+1))/data$(($i+1)).bag" \
            --topics /cloud_registered_body:PointCloud2 /Odometry:Odometry \
            --output_topics /cloud_registered_body:$out_pctopic /Odometry:$out_odomtopic &
    
    # 停顿等待一段时间
    sleep 2

    # 播放对应的bag文件
    rosbag play "${bags[$i]}" -r 1 -s 0 -q &

    # 等待上一个循环的任务完成
    wait
done
