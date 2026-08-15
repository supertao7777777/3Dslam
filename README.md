<div align="center">
  <img src="3DslamGIF.gif" alt="3D Slam demo" width="80%"/>
</div>

# 3D Slam —— 基于 FAST-LIO-SAM 的 LiDAR-IMU 建图与定位系统

## 部署环境

| 依赖 | 版本/说明 |
|------|-----------|
| Ubuntu | 20.04 |
| ROS | Noetic（ROS1） |
| C++ 标准 | C++17（建图/定位包）、C++11（livox 驱动） |
| PCL | 1.10 ⚠️ 需安装在 `/usr/local`（见下方说明） |
| Eigen | 3 |
| GTSAM | ≥ 4.0（仅建图包） |
| Ceres Solver | ≥ 1.14（仅定位包） |
| OpenCV | 3/4 |
| Boost | system filesystem thread chrono timer serialization date_time |
| GeographicLib | 含 CMake 配置文件（`/usr/share/cmake/geographiclib`） |
| OpenMP | 建图/定位包均需要 |
| Python | 3.x（matplotlibcpp 绘图用） |
| 传感器 | Livox Mid-360（驱动已内置，v2.6.0） |

基础依赖安装：

```bash
sudo apt install ros-noetic-pcl-ros libeigen3-dev libopencv-dev libboost-all-dev \
  libceres-dev libgeographic-dev
# GTSAM 官方 apt 源未收录，从 PPA 或源码安装：https://gtsam.org/get_started/
```

⚠️ **PCL 路径注意**：`FAST_LIO_SAM/CMakeLists.txt` 硬编码了 PCL 1.10 的 `/usr/local` 安装路径（`src/FAST_LIO_SAM/CMakeLists.txt:18-20`）。若 PCL 在 `/usr/lib`（apt 默认位置），请二选一：

- 将 PCL 1.10 编译安装到 `/usr/local`，或
- 修改 `FAST_LIO_SAM/CMakeLists.txt` 去掉硬编码，改用 `find_package(PCL)` 自动查找

编译：

```bash
cd ~/ws_3D
catkin_make -j$(nproc)
source devel/setup.bash
```

## 启动

### 建图

```bash
roslaunch fast_lio_sam mapping_mid360.launch        # 实车 Mid-360
roslaunch fast_lio_sam mapping_sim_mid360.launch    # 仿真（点云话题 /scan）
```

另开终端回放数据包：

```bash
rosbag play xxx.bag
```

### 保存地图 / 关键帧 / 二维码

```bash
rosservice call /save_pose "resolution: 0.0
destination: ''"

rosservice call /save_map "resolution: 0.0
destination: ''"

rosservice call /fast_lio_sam/save_qrcode "resolution: 0.0
destination: ''"
```

### 定位

```bash
roslaunch lio_localization run_loc.launch           # 实车定位
roslaunch lio_localization sim_run_loc.launch       # 仿真定位（use_sim_time=true）
```

启动后在 rviz 中用 `2D Pose Estimate` 给定初始位姿，定位即开始。
