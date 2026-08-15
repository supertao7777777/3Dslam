<div align="center">
  <img src="3DslamGIF.gif" alt="3D Slam demo" width="80%"/>
</div>

# 3D Slam —— 基于 FAST-LIO-SAM 的 LiDAR-IMU 建图与定位系统

基于 [FAST-LIO-SAM](https://github.com/engcang/FAST-LIO-SAM) 的 3D 激光 SLAM 系统，面向 Livox Mid-360。包含**建图**（FAST-LIO2 紧耦合里程计 + GTSAM 全局图优化）、**先验地图重定位**（Ceres 后端）与**二维码地标**三个模块，支持实车与仿真两种模式。

## 部署环境

| 依赖 | 版本/说明 |
|------|-----------|
| Ubuntu | 20.04 |
| ROS | Noetic（ROS1） |
| C++ 标准 | C++17（建图/定位包）、C++11（livox 驱动） |
| PCL | 1.10 ⚠️ 需安装在 `/usr/local`（见下方说明） |
| Eigen | 3 |
| GTSAM | ≥ 4.0（仅建图包，官方 PPA 或源码编译） |
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
# GTSAM 官方 apt 源未收录，从 PPA 或源码安装：
# https://gtsam.org/get_started/
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

> livox 驱动依赖 `liblivox_sdk_static.a`（`/usr/local/lib`），缺失时 CMake 会自动 clone Livox-SDK 源码编译，首次编译较慢属正常。

## 主要功能与命令行

### 1. 建图（FAST-LIO-SAM）

```bash
source devel/setup.bash
roslaunch fast_lio_sam mapping_mid360.launch   # 实车 Mid-360
roslaunch fast_lio_sam mapping_sim_mid360.launch  # 仿真（点云话题 /scan）
roslaunch fast_lio_sam mapping_velodyne.launch    # Velodyne 16 线
```

另开终端回放数据包：

```bash
rosbag play xxx.bag
```

### 2. 保存地图 / 关键帧 / 二维码

建图完成后依次调用（`destination` 留空时保存到 `~/ws_3D/src/FAST_LIO_SAM/result/`）：

```bash
rosservice call /save_pose "resolution: 0.0
destination: ''"

rosservice call /save_map "resolution: 0.0
destination: ''"

rosservice call /fast_lio_sam/save_qrcode "resolution: 0.0
destination: ''"
```

### 3. 定位（先验地图重定位）

```bash
source devel/setup.bash
roslaunch lio_localization run_loc.launch      # 实车定位
roslaunch lio_localization sim_run_loc.launch  # 仿真定位（use_sim_time=true）
```

**启动后在 rviz 中用 `2D Pose Estimate` 给定初始位姿**，定位即开始。先验地图从 `src/FAST_LIO_SAM/result/` 加载（`trajectory.pcd`、`CornerMap.pcd`、`SurfMap.pcd`、`GlobalMap.pcd`），二维码地图从 `~/qrcode_poses.txt` 加载。

### 4. 二维码地标

```bash
roslaunch landmark landmark.launch   # TCP 服务端，接收外置二维码相机的检测结果
```

地标观测以因子形式加入建图优化（GTSAM BetweenFactor）与定位约束（位置权重 3000 / 姿态权重 1000，含外点剔除）。

### 5. 雷达驱动（真机时）

```bash
roslaunch livox_ros_driver livox_lidar.launch
```

⚠️ 首次使用需修改 `livox_ros_driver/config/livox_lidar_config.json` 中的广播码（`enable_connect` 设为 true）。

## 输出文件

| 命令 | 输出（`result/` 目录下） |
|------|--------------------------|
| `/save_map` | `GlobalMap.pcd`、`CornerMap.pcd`、`SurfMap.pcd`、`trajectory.pcd`、`transformations.pcd`、`filterGlobalMap.pcd` |
| `/save_pose` | `optimized_pose.txt`、`without_optimized_pose.txt`（KITTI 格式，可喂 evo）、`std_pose.txt`（TUM 格式）、`gnss_pose.txt` |
| `/fast_lio_sam/save_qrcode` | `~/qrcode_poses.txt` |
| 定位过程 | `Lidar_IMU_Localization/Log/odom_trajectory_TUM.txt` |

## 目录结构

```
ws_3D/
├── src/
│   ├── FAST_LIO_SAM/            # 建图：FAST-LIO2 + GTSAM 全局优化（回环/GPS/二维码因子）
│   ├── Lidar_IMU_Localization/  # 定位：特征提取 + 先验地图约束重定位（Ceres）
│   ├── landmark/                # 二维码地标：TCP 协议解析节点
│   └── livox_ros_driver/        # Livox Mid-360 驱动（v2.6.0）
├── 3DslamGIF.gif
└── README.md
```

## 注意事项

- `Lidar_IMU_Localization/launch/run.launch` 的 `project` 参数默认值是旧包名 `LIO_Localization`，直接跑会失败，需加 `project:=lio_localization`
- `/fast_lio_sam/save_qrcode` 实际使用 `save_pose` 消息类型，调用格式与 `/save_pose` 相同
- `FAST_LIO_SAM/launch/` 下多数 launch 引用的 yaml 已缺失，实际维护的只有 mid360 / sim_mid360 / velodyne 三套配置
- 定位支持 `IMU_Mode` 0/1/2（不用 IMU / 松耦合 / 紧耦合），配置见 `Lidar_IMU_Localization/config/mid360.yaml`

## TODO

- [ ] 外推器
- [ ] 地图扩建
- [ ] 定位精度评估与异常检测（来自定位包 README）
- [ ] 编码器融合

## 致谢

- [FAST-LIO-SAM](https://github.com/engcang/FAST-LIO-SAM) —— 建图核心
- [FAST-LIO2](https://github.com/hku-mars/FAST_LIO) —— LiDAR-IMU 里程计前端
- [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM) —— 全局优化后端
- [Livox SDK](https://github.com/Livox-SDK) —— 激光雷达驱动
