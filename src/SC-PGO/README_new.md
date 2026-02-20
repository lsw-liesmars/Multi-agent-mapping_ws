# SC-PGO  ---> 实现多机定位建图

## 环境要求
- **操作系统**：Ubuntu 20.04 
- **ROS 版本**：Melodic 或 Noetic
- **依赖库**：
  - Ceres Solver
  - GTSAM
  - Open3D
  - Eigen

## 编译说明

**配置工作空间并编译**：
   ```bash
   cd ~/Multi-agent-mapping_ws
   # 如果 Ceres 和 GTSAM 安装在系统默认路径（/usr/local），可直接编译
   catkin build
   
   # 或显式指定路径（如果安装在自定义位置）
   # catkin build -DCeres_DIR=/usr/local/lib/cmake/Ceres -DGTSAM_DIR=/usr/local/lib/cmake/GTSAM
   ```

## 文件说明

- **Shell 脚本**：
  - `xxx_snail.sh`：snail radar 数据运行
  - `xxx_Egypt.sh`：中埃重点研发数据运行

- **Launch 文件**：
  - `aloam_mid360_multirobot_whu_egypt.launch`：中埃重点研发数据设置
  - `aloam_mid360_multirobot_whu.launch`：校园自采数据设置

## 数据模型
- **输入数据**：需要多个机器人前端里程计 bag 文件，结构如下：
    topics:      /robot_1/full_cloud   2957 msgs    : sensor_msgs/PointCloud2
                /robot_1/odom         2957 msgs    : nav_msgs/Odometry


- **深度模型**：用于闭环检测和多机间检索：
  - `inhouse_best_model_RI_V6.pt`


