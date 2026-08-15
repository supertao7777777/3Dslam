#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <deque>
#include <string>
#include <thread>
#include <vector>
#include <array>
#include <chrono>
#include <fstream>
#include <sstream>
#include <map>
#include <cmath>
//  ros
#include <ros/ros.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

//  PCL
#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>

//  Eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>

// GPS
#include <sensor_msgs/NavSatFix.h>
#include <GeographicLib/LocalCartesian.hpp>

#include "tool_color_printf.hpp"
#include "mutexDeque.hpp"
#include "tictoc.hpp"
#include "Estimator/Map_Manager.h"
#include "Estimator/ceresfunc.h"

#include "my_utility.h"

//QRcode
#include <lio_localization/QRcode.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <sophus/so3.hpp>

std::string root_dir = ROOT_DIR;

struct PointXYZIRPYT
{
  PCL_ADD_POINT4D
  PCL_ADD_INTENSITY;
  float roll;
  float pitch;
  float yaw;
  double time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRPYT,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, roll, roll)(float, pitch, pitch)(float, yaw, yaw)(double, time, time))

// typedef pcl::PointXYZI PointType;
typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> CLOUD;
typedef CLOUD::Ptr CLOUD_PTR;
typedef PointXYZIRPYT PointTypePose;

struct pcdmap
{
  std::vector<CLOUD_PTR> corner_keyframes_;
  std::vector<CLOUD_PTR> surf_keyframes_;
  CLOUD_PTR globalMapCloud_;
  CLOUD_PTR cloudKeyPoses3D_;
  CLOUD_PTR globalCornerMapCloud_;
  CLOUD_PTR globalSurfMapCloud_;
  pcdmap()
  {
    globalMapCloud_.reset(new CLOUD);
    cloudKeyPoses3D_.reset(new CLOUD);
    globalCornerMapCloud_.reset(new CLOUD);
    globalSurfMapCloud_.reset(new CLOUD);
  }
};

struct kylidar
{
  CLOUD_PTR corner;
  CLOUD_PTR surf;
  CLOUD_PTR cloud;
  double time;
  kylidar() : time(-1)
  {
    corner.reset(new CLOUD);
    surf.reset(new CLOUD);
    cloud.reset(new CLOUD());
  }
};

enum InitializedFlag
{
  NonInitialized,
  Initializing,
  Initialized,
  MayLost
};

/** \brief point to line feature */
struct FeatureLine
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d pointOri;
  Eigen::Vector3d lineP1;
  Eigen::Vector3d lineP2;
  double error;
  bool valid;
  FeatureLine(Eigen::Vector3d po, Eigen::Vector3d p1, Eigen::Vector3d p2)
      : pointOri(std::move(po)), lineP1(std::move(p1)), lineP2(std::move(p2))
  {
    valid = false;
    error = 0;
  }
  double ComputeError(const Eigen::Matrix4d &pose)
  {
    Eigen::Vector3d P_to_Map = pose.topLeftCorner(3, 3) * pointOri + pose.topRightCorner(3, 1);
    double l12 = std::sqrt((lineP1(0) - lineP2(0)) * (lineP1(0) - lineP2(0)) + (lineP1(1) - lineP2(1)) * (lineP1(1) - lineP2(1)) + (lineP1(2) - lineP2(2)) * (lineP1(2) - lineP2(2)));
    double a012 = std::sqrt(
        ((P_to_Map(0) - lineP1(0)) * (P_to_Map(1) - lineP2(1)) - (P_to_Map(0) - lineP2(0)) * (P_to_Map(1) - lineP1(1))) * ((P_to_Map(0) - lineP1(0)) * (P_to_Map(1) - lineP2(1)) - (P_to_Map(0) - lineP2(0)) * (P_to_Map(1) - lineP1(1))) + ((P_to_Map(0) - lineP1(0)) * (P_to_Map(2) - lineP2(2)) - (P_to_Map(0) - lineP2(0)) * (P_to_Map(2) - lineP1(2))) * ((P_to_Map(0) - lineP1(0)) * (P_to_Map(2) - lineP2(2)) - (P_to_Map(0) - lineP2(0)) * (P_to_Map(2) - lineP1(2))) + ((P_to_Map(1) - lineP1(1)) * (P_to_Map(2) - lineP2(2)) - (P_to_Map(1) - lineP2(1)) * (P_to_Map(2) - lineP1(2))) * ((P_to_Map(1) - lineP1(1)) * (P_to_Map(2) - lineP2(2)) - (P_to_Map(1) - lineP2(1)) * (P_to_Map(2) - lineP1(2))));
    error = a012 / l12;

    return error;
  }
};

/** \brief point to plan feature */
struct FeaturePlanVec
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d pointOri;
  Eigen::Vector3d pointProj;
  Eigen::Matrix3d sqrt_info;
  double error;
  bool valid;
  FeaturePlanVec(const Eigen::Vector3d &po, const Eigen::Vector3d &p_proj, Eigen::Matrix3d sqrt_info_)
      : pointOri(po), pointProj(p_proj), sqrt_info(sqrt_info_)
  {
    valid = false;
    error = 0;
  }
  double ComputeError(const Eigen::Matrix4d &pose)
  {
    
    Eigen::Vector3d P_to_Map = pose.topLeftCorner(3, 3) * pointOri + pose.topRightCorner(3, 1);
    error = (P_to_Map - pointProj).norm();
    return error;
  }
};

class map_location
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  struct LidarFrame
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CLOUD_PTR laserCloud;
    CLOUD_PTR corner;
    CLOUD_PTR surf;
    IMUIntegrator imuIntegrator;
    Eigen::Vector3d P;
    Eigen::Vector3d V;
    Eigen::Quaterniond Q;
    Eigen::Vector3d bg;
    Eigen::Vector3d ba;
    double timeStamp;
    LidarFrame()
    {
      corner.reset(new CLOUD);
      surf.reset(new CLOUD);
      laserCloud.reset(new CLOUD());
      P.setZero();
      V.setZero();
      Q.setIdentity();
      bg.setZero();
      ba.setZero();
      timeStamp = 0;
    }
  };

private:
  ros::NodeHandle nh_;
  ros::Subscriber sub_cloud_;
  ros::Subscriber sub_imu_;
  ros::Subscriber sub_initial_pose_;
  ros::Subscriber sub_QRcode ;

  ros::Publisher pub_corner_map;
  ros::Publisher pub_surf_;
  ros::Publisher pubMappedPoints_;
  ros::Publisher pubLaserOdometryPath_;
  ros::Publisher pubOdomImu_;

  tf::StampedTransform transform_;
  tf::TransformBroadcaster broadcaster_; //  publish laser to map tf

  nav_msgs::Path laserOdoPath;

    std::mutex line_feature_mutex_;   // 专门用于线特征
    std::mutex plan_feature_mutex_;   // 专门用于面特征
    std::mutex map_update_mutex_;     // 用于地图更新
    std::mutex extrap_mutex_;

  pcdmap map;
  std::string filename;
  std::string pointCloudTopic;
  std::string imu_topic;
  std::string gps_topic_;
  std::string QRcode_topic;
  int IMU_Mode = 0;
  bool use_lio = false;
  bool publish_downsampled_global_map_ = false;
  bool use_gps_ = false;
  bool gps_use_alt_ = true;
  double gps_cov_min_xy_ = 1.0;   // 平面协方差下限（防止奇异）
  double gps_cov_z_scale_ = 5.0;  // 高程协方差放大倍数
  double gps_huber_width_ = 1.0;  // GPS Huber核宽度
  double gps_time_max_gap_ = 0.5; // GPS 与 lidar 时间最大允许差（秒）
  double corner_leaf_;
  double surf_leaf_;
  double publish_leaf_size_ = 0.0;
  double surround_search_radius_;

  ros::Subscriber sub_gps_;

  struct GpsMeas
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double t;
    Eigen::Vector3d enu;
    Eigen::Matrix3d sqrt_info;
    GpsMeas() : t(0.0), enu(Eigen::Vector3d::Zero()), sqrt_info(Eigen::Matrix3d::Identity()) {}
  };
  std::deque<GpsMeas> gps_queue_;
  std::mutex m_gps_;
  GeographicLib::LocalCartesian geo_converter_;
  bool geo_inited_ = false;

  pcl::KdTreeFLANN<PointType>::Ptr kdtree_keyposes_3d_;
  pcl::KdTreeFLANN<PointType>::Ptr kdtree_corner_map;
  pcl::KdTreeFLANN<PointType>::Ptr kdtree_surf_map;

  pcl::KdTreeFLANN<PointType>::Ptr kdtree_corner_localmap;
  pcl::KdTreeFLANN<PointType>::Ptr kdtree_surf_localmap;

  ros::Publisher pub_surround_corner_;  // 新增：发布周围角点
  ros::Publisher pub_surround_surf_;    // 新增：发布周围面点

  ros::Publisher pub_icp_source_;      // 新增：ICP源点云（待配准点云）
  ros::Publisher pub_icp_target_;      // 新增：ICP目标点云（地图点云）
  ros::Publisher pub_icp_result_;      // 新增：ICP结果点云

  CLOUD_PTR surround_surf;
  CLOUD_PTR surround_corner;

  CLOUD_PTR laserCloudFullRes;

  pcl::VoxelGrid<PointType> ds_corner_;
  pcl::VoxelGrid<PointType> ds_surf_;

  MutexDeque<sensor_msgs::PointCloud2ConstPtr> _lidarMsgQueue;
  // MutexDeque<livox_ros_driver::CustomMsgConstPtr> _lidarMsgQueue;

  MutexDeque<sensor_msgs::ImuConstPtr> _imuMsgQueue;
  InitializedFlag initializedFlag;

  PointTypePose initpose;

  Eigen::Vector3d GravityVector;

  int pushCount = 0;
  double startTime = 0;
  int WINDOWSIZE;
  bool LidarIMUInited = false;
  boost::shared_ptr<std::list<LidarFrame>> lidarFrameList;
    std::string root_dir = ROOT_DIR;

  MAP_MANAGER *map_manager;
  static const int SLIDEWINDOWSIZE = 2;
  double para_PR[SLIDEWINDOWSIZE][6];
  double para_VBias[SLIDEWINDOWSIZE][9];
  MarginalizationInfo *last_marginalization_info = nullptr;
  std::vector<double *> last_marginalization_parameter_blocks;

  Eigen::Matrix4d exTlb = Eigen::Matrix4d::Identity();
  double thres_dist = 1.0;
  double plan_weight_tan = 0.0;
  Eigen::Matrix3d delta_Rl = Eigen::Matrix3d::Identity();
  Eigen::Vector3d delta_tl = Eigen::Vector3d::Zero();
  Eigen::Matrix4d transformLastMapped = Eigen::Matrix4d::Identity();

  bool have_lidar_state_ = false;
  double last_lidar_time_ = -1.0;
  double last_imu_time_ = -1.0;
  Eigen::Vector3d last_lidar_p_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond last_lidar_q_ = Eigen::Quaterniond::Identity();
  Eigen::Vector3d last_lidar_v_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d imu_extrap_p_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond imu_extrap_q_ = Eigen::Quaterniond::Identity();

  static const int localMapWindowSize = 30;
  int localMapID = 0;
  pcl::PointCloud<PointType>::Ptr localCornerMap[localMapWindowSize];
  pcl::PointCloud<PointType>::Ptr localSurfMap[localMapWindowSize];
  pcl::PointCloud<PointType>::Ptr laserCloudCornerFromLocal;
  pcl::PointCloud<PointType>::Ptr laserCloudSurfFromLocal;

  //QRcode
  typedef lio_localization::QRcode QRcode;
  QRcode QRcode_msg;
  ros::Time QRtime_in_saved = ros::Time(0);
  std::uint32_t QR_number_saved = 0;
  std::vector<float> camera_robot_offset; 
  // 相机坐标 -> 机器人中心坐标的外参 T_br_bc（camera to robot）
  Eigen::Matrix4d T_cam_to_robot_ = Eigen::Matrix4d::Identity();
  bool has_T_cam_to_robot_ = false;
  bool if_QRin = false;

  struct QRCodeMapPose {
  int tag_id;
  Eigen::Vector3d position;       // 位置 (x, y, z)
  Eigen::Quaterniond orientation; // 姿态 (qx, qy, qz, qw)
      
  // 必须添加这个宏来确保内存对齐
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      
  QRCodeMapPose() : tag_id(-1) {
  // 显式初始化Eigen成员
  position = Eigen::Vector3d::Zero();
  orientation = Eigen::Quaterniond::Identity();
  }
      
  // 带参数的构造函数
  QRCodeMapPose(int id, double x, double y, double z, 
  double qx, double qy, double qz, double qw)
  : tag_id(id), position(x, y, z), orientation(qw, qx, qy, qz) 
  {
          // 确保四元数归一化
  if (orientation.norm() > 0) {
  orientation.normalize();
  }
      }
      
  // 禁止复制构造函数和赋值运算符（防止对齐问题）
  QRCodeMapPose(const QRCodeMapPose& other) {
  tag_id = other.tag_id;
  position = other.position;
  orientation = other.orientation;
  }
      
  QRCodeMapPose& operator=(const QRCodeMapPose& other) {
  if (this != &other) {
  tag_id = other.tag_id;
  position = other.position;
  orientation = other.orientation;
  }
          return *this;
  }
      
  // 转换为变换矩阵
  Eigen::Matrix4d toMatrix() const {
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3,3>(0,0) = orientation.toRotationMatrix();
  T.block<3,1>(0,3) = position;
  return T;
  }
      
  void print() const {
  std::cout << "QR Code Tag " << tag_id << ": "
  << "pos=[" << position.x() << ", " 
  << position.y() << ", " << position.z() << "], "
  << "quat=[" << orientation.x() << ", "
  << orientation.y() << ", " << orientation.z() << ", "
  << orientation.w() << "]" << std::endl;
  }
  };

  // 使用Eigen对齐分配器的map类型
  using QRCodeMap = std::map<int, QRCodeMapPose, 
  std::less<int>,
  Eigen::aligned_allocator<std::pair<const int, QRCodeMapPose>>>;
    
  QRCodeMap qrcode_map_poses_;
  
  // 二维码位姿文件路径
  std::string qrcode_pose_file_;

  // 二维码可视化发布器
  ros::Publisher pub_qrcode_markers_;
  ros::Timer qrcode_markers_timer_;
  
struct SimpleQRCodeObservation {
        int tag_id;
        Eigen::Vector3d position;      // 二维码在机器人坐标系下的观测位置
        Eigen::Quaterniond orientation; // 二维码在机器人坐标系下的观测姿态
        double timestamp;              // 观测时间戳
        bool valid;                    // 观测是否有效
        
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
        SimpleQRCodeObservation() : tag_id(-1), valid(false), timestamp(0) {
            position.setZero();
            orientation.setIdentity();
        }
    };
    
    // 简化的二维码观测（同一时刻只有一个）
    SimpleQRCodeObservation current_qrcode_obs_;
    
    // 二维码约束权重参数（可在配置文件中调整）
    double qrcode_position_weight_ ;
    double qrcode_orientation_weight_ ;

    // 二维码外点判定阈值与开关（位置 / 角度）
    double qrcode_outlier_position_thresh_;     // m
    double qrcode_outlier_yaw_thresh_deg_;      // deg
    bool   qrcode_reject_outlier_;              // 是否直接丢弃外点二维码

    // 记录最近一次二维码是否被接受（用于在边缘化阶段把信息打包进先验）
    bool   last_qrcode_accepted_;

    private:
  // ========== 简化的二维码残差函数（仅平面位置 + 偏航） ==========
  struct QRCodeCostFunction {
      QRCodeCostFunction(const Eigen::Vector3d& obs_position,
                        const Eigen::Quaterniond& obs_orientation,
                        const Eigen::Vector3d& map_position,
                        const Eigen::Quaterniond& map_orientation,
                        double position_weight = 1.0,
                        double orientation_weight = 1.0)
          : obs_position_(obs_position), 
            obs_orientation_(obs_orientation),
            map_position_(map_position),
            map_orientation_(map_orientation),
            position_weight_(position_weight),
            orientation_weight_(orientation_weight) {}
      
      template <typename T>
      bool operator()(const T* const pose, T* residuals) const {
          // pose: [tx, ty, tz, qx, qy, qz, qw] 或 [tx, ty, tz, rx, ry, rz]
          // 我们需要确定参数化的格式
          
          // 假设pose是 [tx, ty, tz, rx, ry, rz] (位置 + 旋转向量)
          Eigen::Map<const Eigen::Matrix<T, 3, 1>> translation(pose);
          Eigen::Map<const Eigen::Matrix<T, 3, 1>> rotation_vec(pose + 3);
          
          // 将旋转向量转换为旋转矩阵（使用罗德里格斯公式）
          T theta = rotation_vec.norm();
          Eigen::Matrix<T, 3, 3> R;
          if (theta < T(1e-6)) {
              R = Eigen::Matrix<T, 3, 3>::Identity();
          } else {
              Eigen::Matrix<T, 3, 1> axis = rotation_vec / theta;
              T c = cos(theta);
              T s = sin(theta);
              T t = T(1) - c;
              
              T x = axis(0), y = axis(1), z = axis(2);
              R(0,0) = c + x*x*t;   R(0,1) = x*y*t - z*s; R(0,2) = x*z*t + y*s;
              R(1,0) = y*x*t + z*s; R(1,1) = c + y*y*t;   R(1,2) = y*z*t - x*s;
              R(2,0) = z*x*t - y*s; R(2,1) = z*y*t + x*s; R(2,2) = c + z*z*t;
          }
          
          // ========== 位置残差 ==========
          Eigen::Matrix<T, 3, 1> obs_world = R * obs_position_.cast<T>() + translation;
          Eigen::Matrix<T, 2, 1> position_error_xy =
              (obs_world - map_position_.cast<T>()).template head<2>();

          // ========== 姿态残差（仅偏航） ==========
          // 观测姿态：从机器人坐标系变换到世界坐标系（完整姿态）
          Eigen::Matrix<T, 3, 3> R_robot_qr_obs = obs_orientation_.cast<T>().toRotationMatrix();
          Eigen::Matrix<T, 3, 3> R_world_qr_obs = R * R_robot_qr_obs;
          
          // 地图中的二维码姿态
          Eigen::Matrix<T, 3, 3> R_world_qr_map = map_orientation_.cast<T>().toRotationMatrix();
          
          // 仅取偏航误差：相对旋转在 Z 轴上的角度
          Eigen::Matrix<T, 3, 3> R_error = R_world_qr_map.transpose() * R_world_qr_obs;
          T yaw_error = atan2(R_error(1,0), R_error(0,0));
          
          // ========== 加权残差 ==========
          residuals[0] = position_weight_ * position_error_xy(0);
          residuals[1] = position_weight_ * position_error_xy(1);
          residuals[2] = orientation_weight_ * yaw_error;
          
          return true;
      }
      
      static ceres::CostFunction* Create(const Eigen::Vector3d& obs_position,
                                        const Eigen::Quaterniond& obs_orientation,
                                        const Eigen::Vector3d& map_position,
                                        const Eigen::Quaterniond& map_orientation,
                                        double position_weight = 1.0,
                                        double orientation_weight = 1.0) {
          return new ceres::AutoDiffCostFunction<QRCodeCostFunction, 3, 6>(
              new QRCodeCostFunction(obs_position, obs_orientation,
                                    map_position, map_orientation,
                                    position_weight, orientation_weight));
      }
      
  private:
      Eigen::Vector3d obs_position_;
      Eigen::Quaterniond obs_orientation_;
      Eigen::Vector3d map_position_;
      Eigen::Quaterniond map_orientation_;
      double position_weight_;
      double orientation_weight_;
  };

  // GPS 平移因子：仅约束 xyz 平移，不约束旋转
  class GPSPositionFactor : public ceres::SizedCostFunction<3, 6>
  {
  public:
    GPSPositionFactor(const Eigen::Vector3d &enu, const Eigen::Matrix3d &sqrt_info)
        : enu_(enu), sqrt_info_(sqrt_info) {}

    bool Evaluate(double const *const *params, double *residuals, double **jacobians) const override
    {
      Eigen::Map<const Eigen::Matrix<double, 6, 1>> pr(params[0]); // [tx ty tz rx ry rz]
      Eigen::Map<Eigen::Vector3d> r(residuals);

      r = sqrt_info_ * (pr.head<3>() - enu_);

      if (jacobians && jacobians[0])
      {
        Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> J(jacobians[0]);
        J.setZero();
        J.block<3, 3>(0, 0) = sqrt_info_;
      }
      return true;
    }

  private:
    Eigen::Vector3d enu_;
    Eigen::Matrix3d sqrt_info_;
  };

  inline ceres::CostFunction *CreateGPSFactor(const Eigen::Vector3d &enu, const Eigen::Matrix3d &sqrt_info)
  {
    return new GPSPositionFactor(enu, sqrt_info);
  }

public:
  map_location()
  {
    camera_robot_offset.resize(6, 0.0);

    nh_.param<std::string>("location/filedir", filename, "");
    nh_.param<std::string>("location/pointCloudTopic", pointCloudTopic, "points_raw");
    nh_.param<std::string>("location/imuTopic", imu_topic, "/livox/imu");
    nh_.param<std::string>("gps/topic", gps_topic_, "/gps/fix");
    nh_.param<int>("location/IMU_Mode", IMU_Mode, 0);
    nh_.param<bool>("location/use_lio", use_lio, false);
    nh_.param<bool>("location/publish_downsampled_global_map", publish_downsampled_global_map_, false);
    nh_.param<double>("location/publish_leaf_size", publish_leaf_size_, 0.0);
    nh_.param<bool>("gps/use_gps", use_gps_, false);
    nh_.param<bool>("gps/use_altitude", gps_use_alt_, true);
    nh_.param<double>("gps/cov_min_xy", gps_cov_min_xy_, 1.0);
    nh_.param<double>("gps/cov_z_scale", gps_cov_z_scale_, 5.0);
    nh_.param<double>("gps/huber_width", gps_huber_width_, 1.0);
    nh_.param<double>("gps/max_time_gap", gps_time_max_gap_, 0.5);
    nh_.param<double>("location/corner_leaf_", corner_leaf_, 0.2);
    nh_.param<double>("location/surf_leaf_", surf_leaf_, 0.5);
    nh_.param<double>("location/surround_search_radius", surround_search_radius_, 5.0);

    // 二维码相关参数读取
    nh_.param<std::string>("QRcode/QRcode_topic", QRcode_topic, "/QRcode");
    nh_.param<std::vector<float>>("QRcode/camera_robot_offset", camera_robot_offset, std::vector<float>(6, 0.0));

    // 解析相机 -> 机器人中心的刚体变换 T_cam_to_robot_
    // camera_robot_offset: [roll, pitch, yaw, x, y, z] (camera to robot)
    if (camera_robot_offset.size() == 6)
    {
      double roll  = static_cast<double>(camera_robot_offset[0]);
      double pitch = static_cast<double>(camera_robot_offset[1]);
      double yaw   = static_cast<double>(camera_robot_offset[2]);
      double x     = static_cast<double>(camera_robot_offset[3]);
      double y     = static_cast<double>(camera_robot_offset[4]);
      double z     = static_cast<double>(camera_robot_offset[5]);/

      Eigen::AngleAxisd Rx(roll,  Eigen::Vector3d::UnitX());
      Eigen::AngleAxisd Ry(pitch, Eigen::Vector3d::UnitY());
      Eigen::AngleAxisd Rz(yaw,   Eigen::Vector3d::UnitZ());
      Eigen::Matrix3d R = (Rz * Ry * Rx).toRotationMatrix();  // RPY: roll->pitch->yaw

      T_cam_to_robot_.setIdentity();
      T_cam_to_robot_.block<3,3>(0,0) = R;
      T_cam_to_robot_.block<3,1>(0,3) = Eigen::Vector3d(x, y, z);

      has_T_cam_to_robot_ = true;

      std::cout << "[QRCode] Loaded camera->robot extrinsic (RPY xyz): "
                << "roll="  << roll  << ", pitch=" << pitch << ", yaw=" << yaw
                << ", x="   << x     << ", y="    << y     << ", z="   << z  << std::endl;
    }
    else
    {
      has_T_cam_to_robot_ = false;
      std::cout << "[QRCode] WARNING: QRcode/camera_robot_offset size != 6, "
                << "camera pose will be treated as robot pose directly." << std::endl;
    }

    nh_.param<std::string>("QRcode/qrcode_pose_file", qrcode_pose_file_, "");

    nh_.param<double>("QRcode/position_weight", qrcode_position_weight_, 1000.0);
    nh_.param<double>("QRcode/orientation_weight", qrcode_orientation_weight_, 80.0);

    // 外点判定相关参数（可在 yaml 中配置）
    nh_.param<double>("QRcode/outlier_position_thresh", qrcode_outlier_position_thresh_, 5.0);
    nh_.param<double>("QRcode/outlier_yaw_thresh_deg", qrcode_outlier_yaw_thresh_deg_, 25.0);
    nh_.param<bool>("QRcode/reject_outlier", qrcode_reject_outlier_, true);

    last_qrcode_accepted_ = false;

    std::cout << "[QRCode] weights: position=" << qrcode_position_weight_
              << ", orientation=" << qrcode_orientation_weight_
              << "; outlier_pos_thresh=" << qrcode_outlier_position_thresh_ << " m"
              << ", outlier_yaw_thresh=" << qrcode_outlier_yaw_thresh_deg_ << " deg"
              << ", reject_outlier=" << (qrcode_reject_outlier_ ? "true" : "false")
              << std::endl;

    // 初始化二维码观测
    current_qrcode_obs_.valid = false;

    sub_cloud_ = nh_.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 50, &map_location::cloudHandler, this);

    if (IMU_Mode > 0)
      sub_imu_ = nh_.subscribe(imu_topic, 2000, &map_location::imu_callback, this);
    if (use_gps_)
    {
      sub_gps_ = nh_.subscribe<sensor_msgs::NavSatFix>(gps_topic_, 200, &map_location::gpsCallback, this);
      std::cout << "[GPS] Subscription enabled on topic " << gps_topic_ << std::endl;
    }
    if (IMU_Mode < 2)
      WINDOWSIZE = 1;
      // IMU_Mode =2 ：imu预积分
    else
      WINDOWSIZE = 20;
    //订阅全局重定位位姿信息
    sub_initial_pose_ = nh_.subscribe<geometry_msgs::PoseWithCovarianceStamped>("/initialpose", 1, &map_location::initialPoseCB, this);

    //订阅二维码
    sub_QRcode = nh_.subscribe(QRcode_topic, 1 ,&map_location::QRcode_cbk,this);                      

    pub_qrcode_markers_ = nh_.advertise<visualization_msgs::MarkerArray>("/qrcode_markers", 1, true);

    pub_corner_map = nh_.advertise<sensor_msgs::PointCloud2>("/global_corner_map", 1);
    pub_surf_ = nh_.advertise<sensor_msgs::PointCloud2>("/surf_registed", 1);

    pubMappedPoints_ = nh_.advertise<sensor_msgs::PointCloud2>("/laser_cloud_mapped", 10);
    pubLaserOdometryPath_ = nh_.advertise<nav_msgs::Path>("/path_mapped", 5);
    pubOdomImu_ = nh_.advertise<nav_msgs::Odometry>("/odom_imu", 50);

    ds_corner_.setLeafSize(corner_leaf_, corner_leaf_, corner_leaf_);
    ds_surf_.setLeafSize(surf_leaf_, surf_leaf_, surf_leaf_);

    pub_surround_corner_ = nh_.advertise<sensor_msgs::PointCloud2>("/surround_corner_points", 1);//no used
    pub_surround_surf_ = nh_.advertise<sensor_msgs::PointCloud2>("/surround_surf_points", 1);  //no used

    // 初始化ICP调试发布器
    pub_icp_source_ = nh_.advertise<sensor_msgs::PointCloud2>("/debug/icp_source", 1);//no used
    pub_icp_target_ = nh_.advertise<sensor_msgs::PointCloud2>("/debug/icp_target", 1);//no used
    pub_icp_result_ = nh_.advertise<sensor_msgs::PointCloud2>("/debug/icp_result", 1);//no used

    if (loadmap())
      std::cout << ANSI_COLOR_GREEN << "load map successful..." << ANSI_COLOR_RESET << std::endl;
    else
    {
      std::cout << ANSI_COLOR_RED_BOLD << "WARN: load map failed." << ANSI_COLOR_RESET << std::endl;
      return;
    }

    // 加载二维码位姿地图
    if (!qrcode_pose_file_.empty()) {
    if (loadQRCodePosesFromFile(qrcode_pose_file_)) {
    std::cout << ANSI_COLOR_GREEN << "Successfully loaded QR code poses from: " 
    << qrcode_pose_file_ << ANSI_COLOR_RESET << std::endl;
    // 加载成功后立即发布一次
    publishQRCodeMarkers();
    } else {
    std::cout << ANSI_COLOR_YELLOW << "WARN: Failed to load QR code poses from: " 
    << qrcode_pose_file_ << ANSI_COLOR_RESET << std::endl;
    }
    } else {
    std::cout << ANSI_COLOR_YELLOW << "No QR code pose file specified." << ANSI_COLOR_RESET << std::endl;
    }

    std::cout << "ROOT_DIR: " << root_dir << std::endl;
    //  create folder
    std::string command = "mkdir -p " + root_dir + "Log";
    system(command.c_str());

    surround_surf.reset(new CLOUD);
    surround_corner.reset(new CLOUD);
    kdtree_keyposes_3d_.reset(new pcl::KdTreeFLANN<PointType>());
    kdtree_keyposes_3d_->setInputCloud(map.cloudKeyPoses3D_); // init 3d-pose kdtree

    kdtree_corner_map.reset(new pcl::KdTreeFLANN<PointType>());
    kdtree_corner_map->setInputCloud(map.globalCornerMapCloud_);
    kdtree_surf_map.reset(new pcl::KdTreeFLANN<PointType>());
    kdtree_surf_map->setInputCloud(map.globalSurfMapCloud_);

    kdtree_corner_localmap.reset(new pcl::KdTreeFLANN<PointType>());
    kdtree_surf_localmap.reset(new pcl::KdTreeFLANN<PointType>());

    while(pub_corner_map.getNumSubscribers() == 0){
      ros::Duration(0.1).sleep(); // 等待0.1秒
    }
      sensor_msgs::PointCloud2 msg_corner_target;
      // 选择源地图
      CLOUD_PTR map_to_pub = publish_downsampled_global_map_
                                 ? map.globalMapCloud_
                                 : map.globalCornerMapCloud_;
      // 可选再次降采样，发布前保证稀疏度
      if (publish_leaf_size_ > 1e-3)
      {
        pcl::VoxelGrid<PointType> vg_pub;
        vg_pub.setLeafSize(publish_leaf_size_, publish_leaf_size_, publish_leaf_size_);
        CLOUD_PTR tmp(new CLOUD);
        vg_pub.setInputCloud(map_to_pub);
        vg_pub.filter(*tmp);
        map_to_pub = tmp;
        std::cout << "[Map] 发布前额外降采样 leaf=" << publish_leaf_size_
                  << "，点数: " << map_to_pub->size() << std::endl;
      }
      if (publish_downsampled_global_map_)
      {
        std::cout << "[Map] 发布全局降采样地图 GlobalMap.pcd，原始点数: "
                  << map.globalMapCloud_->size() << std::endl;
      }
      else
      {
        std::cout << "[Map] 发布全局角点地图 CornerMap.pcd，原始点数: "
                  << map.globalCornerMapCloud_->size() << std::endl;
      }
      pcl::toROSMsg(*map_to_pub, msg_corner_target);
      msg_corner_target.header.stamp = ros::Time::now();
      msg_corner_target.header.frame_id = "world";
      pub_corner_map.publish(msg_corner_target);

    initializedFlag = NonInitialized;

    for (int i = 0; i < localMapWindowSize; i++)
    {
      localCornerMap[i].reset(new pcl::PointCloud<PointType>);
      localSurfMap[i].reset(new pcl::PointCloud<PointType>);
    }
    laserCloudCornerFromLocal.reset(new pcl::PointCloud<PointType>);
    laserCloudSurfFromLocal.reset(new pcl::PointCloud<PointType>);

    map_manager = new MAP_MANAGER(0.2, 0.3);
    lidarFrameList.reset(new std::list<LidarFrame>);
    
    //设置定时器，每2秒发布一次二维码Marker
    qrcode_markers_timer_ = nh_.createTimer(ros::Duration(2.0),&map_location::publishQRCodeMarkersTimer, this);
  }
  ~map_location();
  //  雷达消息存在_lidarMsgQueue
  void QRcode_cbk(const QRcode &msg_in)
  {
    ros::Time QRtime_in = ros::Time::now();
    //5秒内只获取一次
    if ((QRtime_in - QRtime_in_saved).toSec() > 5.0 ){
      // 原始观测：二维码在相机坐标系下的位姿
      Eigen::Vector3d qr_pos_cam(msg_in.x, msg_in.y, 0.0);
      Eigen::Quaterniond qr_q_cam(Eigen::AngleAxisd(msg_in.yaw, Eigen::Vector3d::UnitZ()));

      // 将观测从相机坐标系变换到机器人中心坐标系
      Eigen::Vector3d qr_pos_robot;
      Eigen::Quaterniond qr_q_robot;
      if (has_T_cam_to_robot_)
      {
        Eigen::Matrix3d R_cam_to_robot = T_cam_to_robot_.block<3,3>(0,0);
        Eigen::Vector3d t_cam_to_robot = T_cam_to_robot_.block<3,1>(0,3);

        qr_pos_robot = R_cam_to_robot * qr_pos_cam + t_cam_to_robot;
        qr_q_robot   = Eigen::Quaterniond(R_cam_to_robot) * qr_q_cam;
      }
      else
      {
        // 未提供外参时，退化为直接认为相机坐标系即机器人坐标系
        qr_pos_robot = qr_pos_cam;
        qr_q_robot   = qr_q_cam;
      }

      // 创建6D观测（二维码在机器人坐标系下的位置和姿态）
      current_qrcode_obs_.tag_id      = msg_in.tag_number;
      current_qrcode_obs_.position    = qr_pos_robot;
      current_qrcode_obs_.orientation = qr_q_robot;
      current_qrcode_obs_.timestamp   = QRtime_in.toSec();
      current_qrcode_obs_.valid       = true;
          
      std::cout << "[QRCode] Detected tag " << msg_in.tag_number 
                << " (camera frame: x=" << msg_in.x << ", y=" << msg_in.y
                << ", yaw=" << msg_in.yaw * 180.0 / M_PI << "°)"
                << " -> robot frame: [" << qr_pos_robot.transpose()
                << "], yaw=" << (Eigen::AngleAxisd(qr_q_robot).angle() * 180.0 / M_PI)
                << "°" << std::endl;

      QRtime_in_saved = QRtime_in;
    }
  }

  void cloudHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
  // void cloudHandler(const livox_ros_driver::CustomMsgConstPtr& msg)    
  {
    if (!_lidarMsgQueue.empty() && (initializedFlag != Initialized)) //  由于tf关系，导致发布时间存在滞后，tf无法显示
      _lidarMsgQueue.clear();
    _lidarMsgQueue.push_back(msg);
  }
  //  imu消息存在_imuMsgQueue
  void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
  {
    // push IMU msg to queue
    _imuMsgQueue.push_back(imu_msg);

    if (IMU_Mode <= 0)
      return;

    double t = imu_msg->header.stamp.toSec();
    if (t <= 0.0)
      return;

    Eigen::Vector3d p;
    Eigen::Quaterniond q;
    Eigen::Vector3d v;

    {
      std::lock_guard<std::mutex> lock(extrap_mutex_);
      if (!have_lidar_state_)
        return;
      if (last_imu_time_ < 0.0)
        last_imu_time_ = t;
      if (t <= last_imu_time_)
        return;

      double dt = t - last_imu_time_;
      Eigen::Vector3d w(imu_msg->angular_velocity.x,
                        imu_msg->angular_velocity.y,
                        imu_msg->angular_velocity.z);
      Eigen::Vector3d wdt = w * dt;
      double angle = wdt.norm();
      Eigen::Quaterniond dq = Eigen::Quaterniond::Identity();
      if (angle > 1e-12)
        dq = Eigen::Quaterniond(Eigen::AngleAxisd(angle, wdt / angle));

      imu_extrap_q_ = (imu_extrap_q_ * dq).normalized();
      imu_extrap_p_ += last_lidar_v_ * dt;
      last_imu_time_ = t;

      p = imu_extrap_p_;
      q = imu_extrap_q_;
      v = last_lidar_v_;
    }

    publishImuOdom(imu_msg->header.stamp, p, q, v);
  }

  void gpsCallback(const sensor_msgs::NavSatFixConstPtr &msg)
  {
    if (!use_gps_)
      return;

    // 基础有效性检查
    if (msg->status.status < sensor_msgs::NavSatStatus::STATUS_FIX ||
        msg->position_covariance_type == sensor_msgs::NavSatFix::COVARIANCE_TYPE_UNKNOWN)
      return;

    // 初始化 ENU 原点
    if (!geo_inited_)
    {
      geo_converter_.Reset(msg->latitude, msg->longitude, msg->altitude);
      geo_inited_ = true;
      std::cout << "[GPS] Set ENU origin: (" << msg->latitude << ", "
                << msg->longitude << ", " << msg->altitude << ")" << std::endl;
    }

    double x, y, z;
    geo_converter_.Forward(msg->latitude, msg->longitude, msg->altitude, x, y, z);

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    cov(0, 0) = msg->position_covariance[0];
    cov(0, 1) = cov(1, 0) = msg->position_covariance[1];
    cov(0, 2) = cov(2, 0) = msg->position_covariance[2];
    cov(1, 1) = msg->position_covariance[4];
    cov(1, 2) = cov(2, 1) = msg->position_covariance[5];
    cov(2, 2) = msg->position_covariance[8];

    // 协方差下限与高度缩放/屏蔽
    cov(0, 0) = std::max(cov(0, 0), gps_cov_min_xy_);
    cov(1, 1) = std::max(cov(1, 1), gps_cov_min_xy_);
    if (gps_use_alt_)
      cov(2, 2) = std::max(cov(2, 2) * gps_cov_z_scale_, gps_cov_min_xy_);
    else
      cov(2, 2) = 1e6; // 近似屏蔽高度

    Eigen::Matrix3d info = cov.inverse();
    Eigen::LLT<Eigen::Matrix3d> info_llt(info);
    if (info_llt.info() != Eigen::Success)
    {
      std::cout << "[GPS] Covariance not PSD, drop measurement." << std::endl;
      return;
    }

    GpsMeas meas;
    meas.t = msg->header.stamp.toSec();
    meas.enu = Eigen::Vector3d(x, y, z);
    meas.sqrt_info = info_llt.matrixL();

    {
      std::lock_guard<std::mutex> lock(m_gps_);
      gps_queue_.push_back(meas);
      if (gps_queue_.size() > 200)
        gps_queue_.pop_front();
    }
  }

  bool fetchClosestGps(double t, GpsMeas &out)
  {
    if (!use_gps_)
      return false;

    std::lock_guard<std::mutex> lock(m_gps_);
    if (gps_queue_.empty())
      return false;

    double best_dt = 1e9;
    size_t best_idx = gps_queue_.size(); // invalid
    for (size_t i = 0; i < gps_queue_.size(); ++i)
    {
      double dt = std::fabs(gps_queue_[i].t - t);
      if (dt < best_dt)
      {
        best_dt = dt;
        best_idx = i;
      }
    }

    if (best_idx == gps_queue_.size() || best_dt > gps_time_max_gap_)
      return false;

    out = gps_queue_[best_idx];
    return true;
  }
  //使用这个函数时,会把特征填入 surf corner
  void ExtractFeature(LidarFrame &kf)
  {
    kf.corner->clear();
    kf.surf->clear();
    for (const auto &p : kf.laserCloud->points)
    {
      if (std::fabs(p.normal_z - 1.0) < 1e-5)
        kf.corner->push_back(p);
    }
    for (const auto &p : kf.laserCloud->points)
    {
      if (std::fabs(p.normal_z - 2.0) < 1e-5)
        kf.surf->push_back(p);
    }
    ds_surf_.setInputCloud(kf.surf);
    ds_surf_.filter(*kf.surf);
    ds_corner_.setInputCloud(kf.corner);
    ds_corner_.filter(*kf.corner);
  }

  // 接受全局定位位置，提取周围特征点用于再配准
  void initialPoseCB(const geometry_msgs::PoseWithCovarianceStampedConstPtr &msg)
  {
    //  低频，不加锁
    PointType p;

    initpose.x = msg->pose.pose.position.x;
    initpose.y = msg->pose.pose.position.y;
    initpose.z = msg->pose.pose.position.z;
    double roll, pitch, yaw;
    tf::Quaternion q;
    tf::quaternionMsgToTF(msg->pose.pose.orientation, q);
    tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
    initpose.roll = roll;
    initpose.pitch = pitch;
    initpose.yaw = yaw;

    p.x = initpose.x;
    p.y = initpose.y;
    p.z = initpose.z;
    std::cout << ANSI_COLOR_RED << "Get initial pose: " << initpose.x << " " << initpose.y << " "
              << initpose.z << " " << roll << " " << pitch << " " << yaw << ANSI_COLOR_RESET << std::endl;
    //提取周围特征点
    extractSurroundKeyFrames(p);
    std::cout << ANSI_COLOR_YELLOW << "Change flat from " << initializedFlag
              << " to " << Initializing << ", start do localizating ..."
              << ANSI_COLOR_RESET << std::endl;
    
    if (initializedFlag != NonInitialized)
    { // TODO: 非第一次执行，需要重置部分参数
      delta_Rl = Eigen::Matrix3d::Identity();
      delta_tl = Eigen::Vector3d::Zero();
      for (int i = 0; i < localMapWindowSize; i++)
      {
        localCornerMap[i]->clear();
        localSurfMap[i]->clear();
      }
      localMapID = 0;
      laserCloudCornerFromLocal->clear();
      laserCloudSurfFromLocal->clear();
    }
    initializedFlag = Initializing;
  }

  void pubOdometry(Eigen::Matrix4d pose, double &time)
  {
    Eigen::Quaterniond Q(pose.block<3, 3>(0, 0));
    tf::Transform transform;

    ros::Time ros_time;
    if (time <= 0) {
        ros_time = ros::Time::now();
        ROS_WARN("Using current time instead of invalid timestamp: %f", time);
    } else {
        ros_time = ros::Time().fromSec(time);
    }

    transform.setOrigin(tf::Vector3(pose(0, 3), pose(1, 3), pose(2, 3)));
    transform.setRotation(tf::Quaternion(Q.x(), Q.y(), Q.z(), Q.w()));
    
    broadcaster_.sendTransform(tf::StampedTransform(transform, ros_time, "world", "base_link"));

    geometry_msgs::PoseStamped laserPose;
    laserPose.header.frame_id = "world";
    laserPose.header.stamp = ros_time;

    laserPose.pose.orientation.x = Q.x();
    laserPose.pose.orientation.y = Q.y();
    laserPose.pose.orientation.z = Q.z();
    laserPose.pose.orientation.w = Q.w();
    laserPose.pose.position.x = pose(0, 3);
    laserPose.pose.position.y = pose(1, 3);
    laserPose.pose.position.z = pose(2, 3);

    laserOdoPath.header.stamp = ros_time;
    laserOdoPath.poses.push_back(laserPose);
    laserOdoPath.header.frame_id = "world";
    pubLaserOdometryPath_.publish(laserOdoPath);

    updateExtrapAnchor(Eigen::Vector3d(pose(0, 3), pose(1, 3), pose(2, 3)), Q, ros_time.toSec());
  }

  void updateExtrapAnchor(const Eigen::Vector3d &p, const Eigen::Quaterniond &q, double time)
  {
    std::lock_guard<std::mutex> lock(extrap_mutex_);
    if (time <= 0.0)
      return;

    if (have_lidar_state_)
    {
      double dt = time - last_lidar_time_;
      if (dt > 1e-3)
        last_lidar_v_ = (p - last_lidar_p_) / dt;
    }

    last_lidar_p_ = p;
    last_lidar_q_ = q;
    last_lidar_time_ = time;
    have_lidar_state_ = true;

    imu_extrap_p_ = p;
    imu_extrap_q_ = q;
    last_imu_time_ = time;
  }

  void publishImuOdom(const ros::Time &stamp, const Eigen::Vector3d &p, const Eigen::Quaterniond &q,
                      const Eigen::Vector3d &v)
  {
    nav_msgs::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "world";
    odom.child_frame_id = "base_link";
    odom.pose.pose.position.x = p.x();
    odom.pose.pose.position.y = p.y();
    odom.pose.pose.position.z = p.z();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    odom.pose.pose.orientation.w = q.w();
    odom.twist.twist.linear.x = v.x();
    odom.twist.twist.linear.y = v.y();
    odom.twist.twist.linear.z = v.z();
    pubOdomImu_.publish(odom);
  }

  void pubOdometry(LidarFrame &frame)
  {
    ros::Time ros_time = ros::Time().fromSec(frame.timeStamp);
    transform_.stamp_ = ros_time;
    transform_.setRotation(tf::Quaternion(frame.Q.x(), frame.Q.y(), frame.Q.z(), frame.Q.w()));
    transform_.setOrigin(tf::Vector3(frame.P(0), frame.P(1), frame.P(2)));
    transform_.frame_id_ = "world";
    transform_.child_frame_id_ = "base_link";
    broadcaster_.sendTransform(transform_);

    geometry_msgs::PoseStamped laserPose;
    laserPose.header.frame_id = "world";
    laserPose.header.stamp = ros_time;

    laserPose.pose.orientation.x = frame.Q.x();
    laserPose.pose.orientation.y = frame.Q.y();
    laserPose.pose.orientation.z = frame.Q.z();
    laserPose.pose.orientation.w = frame.Q.w();
    laserPose.pose.position.x = frame.P(0);
    laserPose.pose.position.y = frame.P(1);
    laserPose.pose.position.z = frame.P(2);

    laserOdoPath.header.stamp = ros_time;
    laserOdoPath.poses.push_back(laserPose);
    // laserOdoPath.header.frame_id = "/world";
    laserOdoPath.header.frame_id = "world";
    pubLaserOdometryPath_.publish(laserOdoPath);

    updateExtrapAnchor(frame.P, frame.Q, frame.timeStamp);
  }

  void vector2double(const std::list<LidarFrame> &tempFrameList)
    {
    int i = 0;
    for (const auto &l : tempFrameList)
    {

      Eigen::Map<Eigen::Matrix<double, 6, 1>> PR(para_PR[i]);
      PR.segment<3>(0) = l.P;
      PR.segment<3>(3) = Sophus::SO3d(l.Q).log();

      Eigen::Map<Eigen::Matrix<double, 9, 1>> VBias(para_VBias[i]);
      VBias.segment<3>(0) = l.V;
      VBias.segment<3>(3) = l.bg;
      VBias.segment<3>(6) = l.ba;
      i++;
    }
  }

  void double2vector(std::list<LidarFrame> &tempFrameList)
    {
    int i = 0;
    for (auto &l : tempFrameList)
    {
      Eigen::Map<const Eigen::Matrix<double, 6, 1>> PR(para_PR[i]);
      Eigen::Map<const Eigen::Matrix<double, 9, 1>> VBias(para_VBias[i]);
      l.P = PR.segment<3>(0);
      l.Q = Sophus::SO3d::exp(PR.segment<3>(3)).unit_quaternion();
      l.V = VBias.segment<3>(0);
      l.bg = VBias.segment<3>(3);
      l.ba = VBias.segment<3>(6);
      i++;
    }
  }
  //lio里程计估计位姿
  void Estimate(std::list<LidarFrame> &frameList, const Eigen::Vector3d &gravity)
    {
      TicToc etc;
      int num_corner_map = 0;
      int num_surf_map = 0;
      int windowSize = frameList.size();
      double t_kd = 0, t_ext = 0;
      etc.tic();
      //局部地图的kd-tree
      if (use_lio)
      {
        //添加缓存的雷达帧
        kdtree_corner_localmap->setInputCloud(laserCloudCornerFromLocal);
        kdtree_surf_localmap->setInputCloud(laserCloudSurfFromLocal);
      }
      t_kd = etc.toc();

      etc.tic();
      //提取frameList中特征
      for (auto &frame : frameList)
        ExtractFeature(frame);
      t_ext = etc.toc();
      std::cout << "make kd-tree: " << t_kd << "ms, extract feature: " << t_ext << "ms" << std::endl;

      // 初始化存放特征的容器
      // store point to line features
      std::vector<std::vector<FeatureLine>> vLineFeatures(windowSize);
      for (auto &v : vLineFeatures)
        v.reserve(2000);

      // store point to plan features
      std::vector<std::vector<FeaturePlanVec>> vPlanFeatures(windowSize);
      for (auto &v : vPlanFeatures)
        v.reserve(2000);

    //搜索距离阈值thres_dist
    if (windowSize == SLIDEWINDOWSIZE)
    {
      plan_weight_tan = 0.0003;
      thres_dist = 1.0;
    }
    else
    {
      plan_weight_tan = 0.0;
      thres_dist = 25.0;
    }

    //optimize process
    const int max_iters = 5;
    int iterOpt = 0;
    for (; iterOpt < max_iters; ++iterOpt)
    {
      double t_search = 0, t_ass = 0, t_solve = 0, t_marg = 0;
      //初始化优化参数
      // std::cout << "frameList.size() = " << frameList.size() << std::endl;
      vector2double(frameList);

      // create huber loss function
      ceres::LossFunction *loss_function = NULL;
      loss_function = new ceres::HuberLoss(0.1 / IMUIntegrator::lidar_m);
      if (windowSize == SLIDEWINDOWSIZE)
      {
        loss_function = NULL;
      }
      else
      {
        loss_function = new ceres::HuberLoss(0.1 / IMUIntegrator::lidar_m);
      }

      //构建优化问题
      ceres::Problem::Options problem_options;
      ceres::Problem problem(problem_options);
      //添加第i帧的位姿 3位置 + 3旋转 需要优化
      for (int i = 0; i < windowSize; ++i)
      {
        problem.AddParameterBlock(para_PR[i], 6);
      }
      //添加速度+bias 3速度 + 3旋转bias 3加速度bias 需要优化
      for (int i = 0; i < windowSize; ++i)
        problem.AddParameterBlock(para_VBias[i], 9);

      // ========== 添加二维码约束 ==========
      // 二维码约束通常很准确，不需要损失函数；只在每次 Estimate 的第一次使用后立刻置 invalid
      if (current_qrcode_obs_.valid)
      {
        // 在地图中查找对应的二维码位姿
        auto map_it = qrcode_map_poses_.find(current_qrcode_obs_.tag_id);

        if (map_it != qrcode_map_poses_.end())
        {
          const auto &map_pose = map_it->second;

          // 使用最新一帧（窗口最后一帧）来计算当前估计下的二维码误差
          int frame_index = windowSize - 1;
          auto frame_iter = frameList.begin();
          std::advance(frame_iter, frame_index);

          Eigen::Matrix4d T_wb = Eigen::Matrix4d::Identity();
          T_wb.block<3, 3>(0, 0) = frame_iter->Q.toRotationMatrix();
          T_wb.block<3, 1>(0, 3) = frame_iter->P;

          // 观测：二维码在机器人坐标系下
          Eigen::Vector3d    obs_pos_robot = current_qrcode_obs_.position;
          Eigen::Quaterniond obs_q_robot   = current_qrcode_obs_.orientation;

          // 预测：二维码在世界坐标系下的位置和姿态
          Eigen::Vector3d obs_pos_world =
              T_wb.block<3, 3>(0, 0) * obs_pos_robot + T_wb.block<3, 1>(0, 3);

          Eigen::Matrix3d R_wb           = T_wb.block<3, 3>(0, 0);
          Eigen::Matrix3d R_robot_qr_obs = obs_q_robot.toRotationMatrix();
          Eigen::Matrix3d R_world_qr_obs = R_wb * R_robot_qr_obs;
          Eigen::Matrix3d R_world_qr_map = map_pose.orientation.toRotationMatrix();

          Eigen::Matrix3d R_error = R_world_qr_map.transpose() * R_world_qr_obs;
          double yaw_error = std::atan2(R_error(1, 0), R_error(0, 0));
          double yaw_error_deg = yaw_error * 180.0 / M_PI;

          Eigen::Vector2d pos_error_xy = (obs_pos_world - map_pose.position).head<2>();

          std::cout << "[QRCode] Adding constraint for tag "
                    << current_qrcode_obs_.tag_id
                    << " to latest frame " << frame_index << std::endl;
          std::cout << "[QRCode] Obs position (robot): "
                    << current_qrcode_obs_.position.transpose() << std::endl;
          std::cout << "[QRCode] Map position (world): "
                    << map_pose.position.transpose() << std::endl;
          std::cout << "[QRCode][Debug] Position error (xy / norm): "
                    << pos_error_xy.transpose() << " / " << pos_error_xy.norm()
                    << " m" << std::endl;
          std::cout << "[QRCode][Debug] Yaw error: "
                    << yaw_error_deg << " deg" << std::endl;

          // ========= 外点检测：用于识别“错误的二维码” =========
          bool is_outlier = false;
          double pos_err_norm = pos_error_xy.norm();

          if (pos_err_norm > qrcode_outlier_position_thresh_ ||
              std::fabs(yaw_error_deg) > qrcode_outlier_yaw_thresh_deg_)
          {
            is_outlier = true;
            std::cout << "[QRCode][Warning] Detected possible WRONG QR code (tag "
                      << current_qrcode_obs_.tag_id << "): pos_error_norm="
                      << pos_err_norm << " m (thres " << qrcode_outlier_position_thresh_
                      << "), yaw_error=" << yaw_error_deg << " deg (thres "
                      << qrcode_outlier_yaw_thresh_deg_ << ")" << std::endl;
          }

          // ========= 根据开关决定是否丢弃外点 =========
          if (is_outlier && qrcode_reject_outlier_)
          {
            std::cout << "[QRCode][Warning] QR constraint is ignored due to outlier."
                      << std::endl;
            last_qrcode_accepted_ = false;
          }
          else
          {
            // 创建并添加仅 XY + 偏航的二维码约束
            ceres::CostFunction *cost_function =
                QRCodeCostFunction::Create(
                    current_qrcode_obs_.position,      // 观测位置（机器人坐标系）
                    current_qrcode_obs_.orientation,   // 观测姿态（机器人坐标系）
                    map_pose.position,                 // 地图位置（世界坐标系）
                    map_pose.orientation,              // 地图姿态（世界坐标系）
                    qrcode_position_weight_,           // 位置权重
                    qrcode_orientation_weight_);       // 姿态权重

            problem.AddResidualBlock(cost_function, nullptr, para_PR[frame_index]);

            if (is_outlier)
            {
              std::cout << "[QRCode] QR constraint is still used "
                        << "(reject_outlier=false)." << std::endl;
            }
            else
            {
              std::cout << "[QRCode] Successfully added QR code constraint" << std::endl;
            }

            last_qrcode_accepted_ = true;
          }
        }
        else
        {
          std::cout << "[QRCode] Warning: Tag " << current_qrcode_obs_.tag_id
                    << " not found in map!" << std::endl;
          last_qrcode_accepted_ = false;
        }

        // 使用后立即置为无效，防止在同一轮 Estimate 里重复使用
        current_qrcode_obs_.valid = false;
      }

      //  TODO: add imu here
      for (int f = 1; f < windowSize; ++f)
      {
        auto frame_curr = frameList.begin();
        std::advance(frame_curr, f);// frame_curr 向前移动f帧

        //这个约束强制要求：从帧f-1到帧f的状态变化必须与IMU预积分测量值一致,即：添加imu预积分约束
        problem.AddResidualBlock(Cost_NavState_PRV_Bias::Create(frame_curr->imuIntegrator,  //创建imu预积分代价函数,当前帧的imu预积分器
                                                                const_cast<Eigen::Vector3d &>(gravity), //重力向量
                                                                Eigen::LLT<Eigen::Matrix<double, 15, 15>>(frame_curr->imuIntegrator.GetCovariance().inverse())
                                                                    .matrixL()
                                                                    .transpose()), //信息矩阵平方根
                                 nullptr,  // 不使用损失函数
                                 para_PR[f - 1], // 前一帧位姿参数
                                 para_VBias[f - 1], // 前一帧速度+偏置参数
                                 para_PR[f], // 当前帧位姿参数
                                 para_VBias[f]); // 当前帧速度+偏置参数
      }

      // GPS 位置约束（仅平移）
      if (use_gps_)
      {
        for (int f = 0; f < windowSize; ++f)
        {
          auto frame_curr = frameList.begin();
          std::advance(frame_curr, f);
          GpsMeas gps_meas;
          if (fetchClosestGps(frame_curr->timeStamp, gps_meas))
          {
            ceres::CostFunction *gps_cost = CreateGPSFactor(gps_meas.enu, gps_meas.sqrt_info);
            ceres::LossFunction *gps_loss = new ceres::HuberLoss(gps_huber_width_);
            problem.AddResidualBlock(gps_cost, gps_loss, para_PR[f]);
          }
        }
      }

      //如果滑动窗口满了,更新先验约束，保存去掉的旧变量信息
      if (last_marginalization_info)
      {
        // construct new marginlization_factor
        // 构造边缘化因子
        auto *marginalization_factor = new MarginalizationFactor(last_marginalization_info);

        //添加边缘化因子
        problem.AddResidualBlock(marginalization_factor, nullptr,
                                 last_marginalization_parameter_blocks);
      }

      // 记录优化前的位姿（用于收敛判断）
      Eigen::Quaterniond q_before_opti = frameList.back().Q;
      Eigen::Vector3d t_before_opti = frameList.back().P;
      
      // 初始化特征关联的容器
      std::vector<std::vector<ceres::CostFunction *>> edgesLine(windowSize);
      std::vector<std::vector<ceres::CostFunction *>> edgesPlan(windowSize);

      // 准备多线程
      std::thread threads[2];

      etc.tic();
      for (int f = 0; f < windowSize; ++f)
      {
        auto frame_curr = frameList.begin();
        std::advance(frame_curr, f);

        //构建当前帧的变换矩阵
        Eigen::Matrix4d transformTobeMapped = Eigen::Matrix4d::Identity();

        transformTobeMapped.topLeftCorner(3, 3) = frame_curr->Q.toRotationMatrix();
        transformTobeMapped.topRightCorner(3, 1) = frame_curr->P;

        //点 线特征
        threads[0] = std::thread(&map_location::processPointToLine, this,
                                 std::ref(edgesLine[f]),              // 输出：线特征代价函数
                                 std::ref(vLineFeatures[f]),          // 输出：线特征数据
                                 std::ref(frame_curr->corner),         // 输入：当前帧角点
                                 std::ref(map.globalCornerMapCloud_),   // 全局角点地图
                                 std::ref(kdtree_corner_map),           // 全局角点KD-tree
                                 std::ref(laserCloudCornerFromLocal),   // 局部角点地图
                                 std::ref(kdtree_corner_localmap),    // 局部角点KD-tree
                                 std::ref(exTlb),                     // 外参：激光到IMU
                                 std::ref(transformTobeMapped));      // 当前估计位姿

        // 点 面特征
        threads[1] = std::thread(&map_location::processPointToPlanVec, this,
                                std::ref(edgesPlan[f]),           // 输出：面特征代价函数
                                std::ref(vPlanFeatures[f]),       // 输出：面特征数据
                                std::ref(frame_curr->surf),       // 输入：当前帧面点
                                std::ref(map.globalSurfMapCloud_), // 全局面点地图
                                std::ref(kdtree_surf_map),        // 全局面点KD-tree
                                std::ref(laserCloudSurfFromLocal), // 局部面点地图
                                std::ref(kdtree_surf_localmap),   // 局部面点KD-tree
                                std::ref(exTlb),                 // 外参：激光到IMU
                                std::ref(transformTobeMapped));  // 当前估计位姿

        threads[0].join();
        threads[1].join();
      }
      t_search = etc.toc();

      int linevaild = 0;
      int sufvaild = 0;

      etc.tic();
      //筛选特征，用于复用
      if (windowSize == SLIDEWINDOWSIZE)
      {
        thres_dist = 1.0;
        if (iterOpt == 0)
        {
          for (int f = 0; f < windowSize; ++f)
          {
            //特征筛选,标记有效特征
            int cntFtu = 0;
            for (auto &e : edgesLine[f])
            {
              if (std::fabs(vLineFeatures[f][cntFtu].error) > 1e-5)
              {
                problem.AddResidualBlock(e, loss_function, para_PR[f]);
                vLineFeatures[f][cntFtu].valid = true;

                linevaild ++;
              }
              else
              {
                vLineFeatures[f][cntFtu].valid = false;
              }
              cntFtu++;
            }

            cntFtu = 0;
            for (auto &e : edgesPlan[f])
            {
              if (std::fabs(vPlanFeatures[f][cntFtu].error) > 1e-5)
              {
                problem.AddResidualBlock(e, loss_function, para_PR[f]);
                vPlanFeatures[f][cntFtu].valid = true;

                sufvaild ++;
              }
              else
              {
                vPlanFeatures[f][cntFtu].valid = false;
              }
              cntFtu++;
            }
          }
        }
        else
        //直接使用第一次迭代时标记的有效valid点
        {
          for (int f = 0; f < windowSize; ++f)
          {
            int cntFtu = 0;
            for (auto &e : edgesLine[f])
            {
              if (vLineFeatures[f][cntFtu].valid)
              {
                problem.AddResidualBlock(e, loss_function, para_PR[f]);
              }
              cntFtu++;
            }
            cntFtu = 0;
            for (auto &e : edgesPlan[f])
            {
              if (vPlanFeatures[f][cntFtu].valid)
              {
                problem.AddResidualBlock(e, loss_function, para_PR[f]);
              }
              cntFtu++;
            }
          }
        }
      }
      else
      {
        if (iterOpt == 0)
        {
          thres_dist = 10.0;
        }
        else
        {
          thres_dist = 1.0;
        }
        for (int f = 0; f < windowSize; ++f)
        {
          int cntFtu = 0;
          for (auto &e : edgesLine[f])
          {
            if (std::fabs(vLineFeatures[f][cntFtu].error) > 1e-5)
            {
              problem.AddResidualBlock(e, loss_function, para_PR[f]);
              vLineFeatures[f][cntFtu].valid = true;
            }
            else
            {
              vLineFeatures[f][cntFtu].valid = false;
            }
            cntFtu++;
          }
          cntFtu = 0;
          for (auto &e : edgesPlan[f])
          {
            if (std::fabs(vPlanFeatures[f][cntFtu].error) > 1e-5)
            {
              problem.AddResidualBlock(e, loss_function, para_PR[f]);
              vPlanFeatures[f][cntFtu].valid = true;
            }
            else
            {
              vPlanFeatures[f][cntFtu].valid = false;
            }
            cntFtu++;
          }
        }
      }
      t_ass = etc.toc();

      //ceres配置
      ceres::Solver::Options options;
      options.linear_solver_type = ceres::DENSE_SCHUR;
      options.trust_region_strategy_type = ceres::DOGLEG;
      options.max_num_iterations = 5;
      options.minimizer_progress_to_stdout = false;
      options.num_threads = 6;
      ceres::Solver::Summary summary;
      etc.tic();
      //参数：怎么解 解什么，解的结果
      ceres::Solve(options, &problem, &summary);
      t_solve = etc.toc();

      double2vector(frameList);

      Eigen::Quaterniond q_after_opti = frameList.back().Q;
      Eigen::Vector3d t_after_opti = frameList.back().P;

      double deltaR = (q_before_opti.angularDistance(q_after_opti)) * 180.0 / M_PI;
      double deltaT = (t_before_opti - t_after_opti).norm();

      //边缘化,将老帧信息变成先验信息
      if (deltaR < 0.05 && deltaT < 0.05 || (iterOpt + 1) == max_iters)
      {
        if (windowSize != SLIDEWINDOWSIZE)
          break;
        etc.tic();
        auto *marginalization_info = new MarginalizationInfo();
        if (last_marginalization_info)
        {
          std::vector<int> drop_set;
          for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
          {
            if (last_marginalization_parameter_blocks[i] == para_PR[0] ||
                last_marginalization_parameter_blocks[i] == para_VBias[0])
              drop_set.push_back(i);
          }

          auto *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
          auto *residual_block_info = new ResidualBlockInfo(marginalization_factor, nullptr,
                                                            last_marginalization_parameter_blocks,
                                                            drop_set);
          marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        auto frame_curr = frameList.begin();
        std::advance(frame_curr, 1);
        ceres::CostFunction *IMU_Cost = Cost_NavState_PRV_Bias::Create(frame_curr->imuIntegrator,
                                                                       const_cast<Eigen::Vector3d &>(gravity),
                                                                       Eigen::LLT<Eigen::Matrix<double, 15, 15>>(frame_curr->imuIntegrator.GetCovariance().inverse())
                                                                           .matrixL()
                                                                           .transpose());
        auto *residual_block_info = new ResidualBlockInfo(IMU_Cost, nullptr,
                                                          std::vector<double *>{para_PR[0], para_VBias[0], para_PR[1], para_VBias[1]},
                                                          std::vector<int>{0, 1});
        marginalization_info->addResidualBlockInfo(residual_block_info);

        int f = 0;
        Eigen::Matrix4d transformTobeMapped = Eigen::Matrix4d::Identity();
        transformTobeMapped.topLeftCorner(3, 3) = frame_curr->Q.toRotationMatrix();
        transformTobeMapped.topRightCorner(3, 1) = frame_curr->P;
        edgesLine[f].clear();
        edgesPlan[f].clear();
        threads[0] = std::thread(&map_location::processPointToLine, this,
                                 std::ref(edgesLine[f]),
                                 std::ref(vLineFeatures[f]),
                                 std::ref(frame_curr->corner),
                                 std::ref(map.globalCornerMapCloud_),
                                 std::ref(kdtree_corner_map),
                                 std::ref(laserCloudCornerFromLocal),
                                 std::ref(kdtree_corner_localmap),
                                 std::ref(exTlb),
                                 std::ref(transformTobeMapped));

        threads[1] = std::thread(&map_location::processPointToPlanVec, this,
                                 std::ref(edgesPlan[f]),
                                 std::ref(vPlanFeatures[f]),
                                 std::ref(frame_curr->surf),
                                 std::ref(map.globalSurfMapCloud_),
                                 std::ref(kdtree_surf_map),
                                 std::ref(laserCloudSurfFromLocal),
                                 std::ref(kdtree_surf_localmap),
                                 std::ref(exTlb),
                                 std::ref(transformTobeMapped));

        threads[0].join();
        threads[1].join();

        int cntFtu = 0;
        for (auto &e : edgesLine[f])
        {
          if (vLineFeatures[f][cntFtu].valid)
          {
            auto *residual_block_info = new ResidualBlockInfo(e, nullptr,
                                                              std::vector<double *>{para_PR[0]},
                                                              std::vector<int>{0});
            marginalization_info->addResidualBlockInfo(residual_block_info);
          }
          cntFtu++;
        }
        cntFtu = 0;
        for (auto &e : edgesPlan[f])
        {
          if (vPlanFeatures[f][cntFtu].valid)
          {
            auto *residual_block_info = new ResidualBlockInfo(e, nullptr,
                                                              std::vector<double *>{para_PR[0]},
                                                              std::vector<int>{0});
            marginalization_info->addResidualBlockInfo(residual_block_info);
          }
          cntFtu++;
        }

        // ========== 将二维码约束也纳入边缘化先验 ==========
        // 说明：
        //  - 这里只在最近一次二维码观测被接受（非外点）时才添加 QR 残差；
        //  - 这里把二维码视为作用在当前滑窗中“被边缘化掉”的那一帧（para_PR[0]) 上的先验约束；
        //  - 通过与 IMU / 点云因子的联合边缘化，其信息会被压缩进对剩余状态的先验中。
        if (last_qrcode_accepted_ && current_qrcode_obs_.tag_id >= 0)
        {
          auto map_it_qr = qrcode_map_poses_.find(current_qrcode_obs_.tag_id);
          if (map_it_qr != qrcode_map_poses_.end())
          {
            const auto &map_pose_qr = map_it_qr->second;

            ceres::CostFunction *qr_cost =
                QRCodeCostFunction::Create(
                    current_qrcode_obs_.position,
                    current_qrcode_obs_.orientation,
                    map_pose_qr.position,
                    map_pose_qr.orientation,
                    qrcode_position_weight_,
                    qrcode_orientation_weight_);

            auto *qr_residual_block_info = new ResidualBlockInfo(
                qr_cost,
                nullptr,
                std::vector<double *>{para_PR[0]},
                std::vector<int>{0});  // 边缘化掉 para_PR[0]

            marginalization_info->addResidualBlockInfo(qr_residual_block_info);

            std::cout << "[QRCode][Marg] Added QR prior for tag "
                      << current_qrcode_obs_.tag_id
                      << " into marginalization." << std::endl;

            // 本次二维码信息已打包进先验，后续不再重复使用
            last_qrcode_accepted_ = false;
          }
        }

        marginalization_info->preMarginalize();
        marginalization_info->marginalize();

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i < SLIDEWINDOWSIZE; i++)
        {
          addr_shift[reinterpret_cast<long>(para_PR[i])] = para_PR[i - 1];
          addr_shift[reinterpret_cast<long>(para_VBias[i])] = para_VBias[i - 1];
        }
        std::vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
        t_marg = etc.toc();
        std::cout << "marg takes:" << t_marg << "ms" << std::endl;
        break;
      }
      if (windowSize != SLIDEWINDOWSIZE)
      {
        for (int f = 0; f < windowSize; ++f)
        {
          edgesLine[f].clear();
          edgesPlan[f].clear();
          vLineFeatures[f].clear();
          vPlanFeatures[f].clear();
        }
      }

      // 原本这里有二维码约束的内存清理，现在约束直接在每次添加时创建，
      // 由 Ceres 在问题销毁时统一管理，这里不再需要额外的 delete。
    }
    // std::cout << "estimate iter: " << iterOpt << std::endl;
  }
  //mid360 unused
  void moveFromCustomMsg_normal(const livox_ros_driver::CustomMsg &Msg, pcl::PointCloud<pcl::PointXYZINormal> & cloud)
  {
      cloud.clear();
      cloud.reserve(Msg.point_num);
      PointType point;

      cloud.header.frame_id=Msg.header.frame_id;
      cloud.header.stamp=Msg.header.stamp.toNSec()/1000;
      cloud.header.seq=Msg.header.seq;

      for(uint i=0;i<Msg.point_num-1;i++)
      {
          point.x=Msg.points[i].x; 
          point.y=Msg.points[i].y; 
          point.z=Msg.points[i].z; 
          point.intensity=Msg.points[i].reflectivity; 
          // point.tag=Msg.points[i].tag; 
          // time 单位是秒
          // point.time=Msg.points[i].offset_time*1e-9; 
          // point.ring=Msg.points[i].line;
          point.normal_y = Msg.points[i].line;
          point.normal_z = 0;
          point.normal_x = (Msg.points[i].offset_time*1e-9 - Msg.points[0].offset_time*1e-9)/(Msg.points[Msg.point_num-1].offset_time*1e-9 - Msg.points[0].offset_time*1e-9);            
          cloud.push_back(point);
      }
  }

  void run()
  {
    double time_last_lidar = -1;
    double time_curr_lidar = -1;
    std::vector<sensor_msgs::ImuConstPtr> vimuMsg;
    //一直循环
    while (true)
    {
      // 等待数据初始化
      if (initializedFlag == NonInitialized)
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }
      if (_lidarMsgQueue.empty())
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }

      //  update frame
      // 更新当前帧
      time_curr_lidar = _lidarMsgQueue.front()->header.stamp.toSec();
      laserCloudFullRes.reset(new CLOUD());
      pcl::fromROSMsg(*_lidarMsgQueue.front(), *laserCloudFullRes);
      // 取一帧
      _lidarMsgQueue.pop_front();
      // ----imu初始化--- 接受雷达帧后
      if (IMU_Mode > 0 && time_last_lidar > 0)
      {
        // get IMU msg int the Specified time interval
        vimuMsg.clear();
        int countFail = 0;

        //找到上帧点云到这帧点云内的imu数据
        while (!fetchImuMsgs(time_last_lidar, time_curr_lidar, vimuMsg))
        {
          countFail++;
          if (countFail > 100)
          {
            if (_imuMsgQueue.empty())
              std::cout << "imu queue is empty." << std::endl;
            else
              std::cout << "imu time: " << _imuMsgQueue.front()->header.stamp.toSec() << "-->" << _imuMsgQueue.back()->header.stamp.toSec() << std::endl;
            break;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      }

      LidarFrame lidarFrame;
      lidarFrame.timeStamp = time_curr_lidar;
      lidarFrame.laserCloud = laserCloudFullRes;
      // lidar_list当前imu guess后的预测帧，叫list是徐晃一枪
      boost::shared_ptr<std::list<LidarFrame>> lidar_list;
      
      if (!vimuMsg.empty())
      {
        if (!LidarIMUInited)
        {
          lidarFrame.imuIntegrator.PushIMUMsg(vimuMsg);
          //旋转积分
          lidarFrame.imuIntegrator.GyroIntegration(time_last_lidar);
          delta_Rl = lidarFrame.imuIntegrator.GetDeltaQ().toRotationMatrix();

          //  predict current lidar pose
          lidarFrame.P = transformLastMapped.topLeftCorner(3, 3) * delta_tl + transformLastMapped.topRightCorner(3, 1);
          Eigen::Matrix3d m3d = transformLastMapped.topLeftCorner(3, 3) * delta_Rl;
          lidarFrame.Q = m3d;

          lidar_list.reset(new std::list<LidarFrame>);
                    lidar_list->push_back(lidarFrame);  

        }
        else
        {
          // if get IMU msg successfully, use pre-integration to update delta lidar pose
          lidarFrame.imuIntegrator.PushIMUMsg(vimuMsg);
          lidarFrame.imuIntegrator.PreIntegration(lidarFrameList->back().timeStamp, lidarFrameList->back().bg, lidarFrameList->back().ba);

          const Eigen::Vector3d &Pwbpre = lidarFrameList->back().P;
          const Eigen::Quaterniond &Qwbpre = lidarFrameList->back().Q;
          const Eigen::Vector3d &Vwbpre = lidarFrameList->back().V;

          const Eigen::Quaterniond &dQ = lidarFrame.imuIntegrator.GetDeltaQ();
          const Eigen::Vector3d &dP = lidarFrame.imuIntegrator.GetDeltaP();
          const Eigen::Vector3d &dV = lidarFrame.imuIntegrator.GetDeltaV();
          double dt = lidarFrame.imuIntegrator.GetDeltaTime();

          lidarFrame.Q = Qwbpre * dQ;
          lidarFrame.P = Pwbpre + Vwbpre * dt + 0.5 * GravityVector * dt * dt + Qwbpre * (dP);
          lidarFrame.V = Vwbpre + GravityVector * dt + Qwbpre * (dV);
          lidarFrame.bg = lidarFrameList->back().bg;
          lidarFrame.ba = lidarFrameList->back().ba;

          // std::cout << "-----bg=" <<  lidarFrame.bg << std::endl;
          // std::cout << "-----ba=" <<  lidarFrame.ba << std::endl;

          Eigen::Quaterniond Qwlpre = Qwbpre;
          Eigen::Vector3d Pwlpre = Pwbpre;

          Eigen::Quaterniond Qwl = lidarFrame.Q;
          Eigen::Vector3d Pwl = lidarFrame.P;

          delta_Rl = Qwlpre.conjugate() * Qwl;
          delta_tl = Qwlpre.conjugate() * (Pwl - Pwlpre);
          // delta_Rb = dQ.toRotationMatrix();
          // delta_tb = dP;

          lidarFrameList->push_back(lidarFrame);
          lidarFrameList->pop_front();
          lidar_list = lidarFrameList;

        }
      }
      else
      {
        if (LidarIMUInited)
          break;
        else
        {
          //  predict pose use constant velocity
          lidarFrame.P = transformLastMapped.topLeftCorner(3, 3) * delta_tl + transformLastMapped.topRightCorner(3, 1);
          Eigen::Matrix3d m3d = transformLastMapped.topLeftCorner(3, 3) * delta_Rl;
          lidarFrame.Q = m3d;

          lidar_list.reset(new std::list<LidarFrame>);
                    lidar_list->push_back(lidarFrame);

        }
      }
      //点云畸变矫正
      RemoveLidarDistortion(laserCloudFullRes, delta_Rl, delta_tl);
      //  publish cloud after remove distort
      sensor_msgs::PointCloud2 laserCloudMsg;
      // livox_ros_driver::CustomMsg laserCloudMsg;

      pcl::toROSMsg(*lidarFrame.laserCloud, laserCloudMsg);
      // laserCloudMsg.header.frame_id = "/base_link";

      laserCloudMsg.header.frame_id = "base_link";
      laserCloudMsg.header.stamp.fromSec(lidarFrame.timeStamp); 
      pubMappedPoints_.publish(laserCloudMsg);

      if (initializedFlag == Initializing)
      {
        bool icp_ok = ICPScanMatchGlobal(*lidar_list);
        Eigen::Matrix4d init_mat = Eigen::Matrix4d::Identity();

        if (icp_ok)
        {
          initializedFlag = Initialized;
          std::cout << ANSI_COLOR_GREEN << "icp scan match successful ..." << ANSI_COLOR_RESET << std::endl;
          init_mat.topLeftCorner(3, 3) = lidar_list->front().Q.toRotationMatrix();
          init_mat.topRightCorner(3, 1) = lidar_list->front().P;
        }
        else
        {
          // fallback: use user-provided initial guess as current pose
          init_mat = toMatrix(initpose);
          lidar_list->front().Q = Eigen::Quaterniond(init_mat.topLeftCorner<3, 3>());
          lidar_list->front().P = init_mat.topRightCorner(3, 1);
          initializedFlag = Initialized;
          std::cout << ANSI_COLOR_YELLOW << "[Init] ICP failed, fallback to initial guess pose." << ANSI_COLOR_RESET << std::endl;
        }

        transformLastMapped = init_mat;
        pubOdometry(transformLastMapped, lidar_list->front().timeStamp);
      }
      else if (initializedFlag == Initialized)
      {
        TicToc tc;
        Eigen::Matrix4d transformAftMapped = Eigen::Matrix4d::Identity();
        double t1, t2, t3;
        //  TODO: 增加局部地图
        int laserCloudCornerFromLocalNum = laserCloudCornerFromLocal->points.size();
        int laserCloudSurfFromLocalNum = laserCloudSurfFromLocal->points.size();
        if ((kdtree_surf_map && kdtree_corner_map) ||
            (laserCloudCornerFromLocalNum > 0 && laserCloudSurfFromLocalNum > 100))
        {
          tc.tic();
          // std::cout<< "lio里程计估计中---" << std::endl;
          // std::cout<< "重力向量---" << GravityVector << std::endl;
          int l=0;
          for (const auto& frame : *lidar_list) {
              // std::cout << "LidarFrame 时间戳: " << frame.timeStamp << std::endl;
              // std::cout << "点云数量: " << frame.laserCloud->size() << std::endl;
              
              // // 如果需要打印每个点的坐标
              // for (const auto& point : *frame.laserCloud) {
              //     std::cout << "点坐标: (" << point.x << ", " << point.y << ", " << point.z << ")" << std::endl;
              // }
              l++;
              // std::cout << "-------------------" << std::endl;
          }
          // std::cout<< "list中点云数量" << l << std::endl;
          Estimate(*lidar_list, GravityVector);

          t1 = tc.toc();
          tc.tic();

          transformAftMapped = Eigen::Matrix4d::Identity();
          transformAftMapped.topLeftCorner(3, 3) = lidar_list->front().Q.toRotationMatrix();
          transformAftMapped.topRightCorner(3, 1) = lidar_list->front().P;
          pubOdometry(transformAftMapped, lidar_list->front().timeStamp);

          Eigen::Matrix3d Rwlpre = transformLastMapped.topLeftCorner(3, 3);
          Eigen::Vector3d Pwlpre = transformLastMapped.topRightCorner(3, 1);
          delta_Rl = Rwlpre.transpose() * transformAftMapped.topLeftCorner(3, 3);
          delta_tl = Rwlpre.transpose() * (transformAftMapped.topRightCorner(3, 1) - Pwlpre);
          transformLastMapped = transformAftMapped;
          t2 = tc.toc();
        }
        tc.tic();
        laserCloudCornerFromLocal->clear();
        laserCloudSurfFromLocal->clear();
        if (use_lio)
          MapIncrementLocal(lidar_list->front());
        t3 = tc.toc();
        std::cout << "LIO takes: " << t1 << ",pubodom:" << t2 << ",mapincrement:" << t3 << std::endl;
  #define SAVE_TRAJ

  #ifdef SAVE_TRAJ
          if (1)
          {
            static std::fstream fout(root_dir + "Log/odom_trajectory_TUM.txt", std::ios::out);
            static std::ostringstream stamp;
            stamp.str("");
            if (fout.is_open())
            {
              // std::string tstamp = to_string(ros::Time().fromSec(laser_odometry->time));
              std::string tstamp = std::to_string(lidar_list->front().timeStamp);
              saveTrajectoryTUMformat(fout, tstamp, lidar_list->front().P(0), lidar_list->front().P(1), lidar_list->front().P(2),
                                      lidar_list->front().Q.x(), lidar_list->front().Q.y(), lidar_list->front().Q.z(), lidar_list->front().Q.w());
            }
          }
  #endif

        // if tightly coupled IMU message, start IMU initialization
        if (IMU_Mode > 1 && !LidarIMUInited)
        {
          // update lidar frame pose
          lidarFrame.P = transformAftMapped.topRightCorner(3, 1);
          Eigen::Matrix3d m3d = transformAftMapped.topLeftCorner(3, 3);
          lidarFrame.Q = m3d;

          // static int pushCount = 0;
          // std::cout << "lidarframelist: " << lidarFrameList->size() << std::endl;
          if (pushCount == 0)
          {
            lidarFrameList->push_back(lidarFrame);
            lidarFrameList->back().imuIntegrator.Reset();
            if (lidarFrameList->size() > WINDOWSIZE)
              lidarFrameList->pop_front();
          }
          else
          {
            lidarFrameList->back().laserCloud = lidarFrame.laserCloud;
            lidarFrameList->back().imuIntegrator.PushIMUMsg(vimuMsg);
            lidarFrameList->back().timeStamp = lidarFrame.timeStamp;
            lidarFrameList->back().P = lidarFrame.P;
            lidarFrameList->back().Q = lidarFrame.Q;
          }
          // std::cout << "lidarframelist: " << lidarFrameList->size() << std::endl;

          pushCount++;
          if (pushCount >= 3)
          {
            pushCount = 0;
            if (lidarFrameList->size() > 1)
            {
              auto iterRight = std::prev(lidarFrameList->end());
              auto iterLeft = std::prev(std::prev(lidarFrameList->end()));
              iterRight->imuIntegrator.PreIntegration(iterLeft->timeStamp, iterLeft->bg, iterLeft->ba);
            }

            if (lidarFrameList->size() == int(WINDOWSIZE / 1.5))
            {
              startTime = lidarFrameList->back().timeStamp;
            }

            if (!LidarIMUInited && lidarFrameList->size() == WINDOWSIZE && lidarFrameList->front().timeStamp >= startTime)
            {
              std::cout << "**************Start MAP Initialization!!!******************" << std::endl;
              if (TryMAPInitialization())
              {
                LidarIMUInited = true;
                pushCount = 0;
                startTime = 0;
              }
              std::cout << "**************Finish MAP Initialization!!!******************" << std::endl;
            }
          }
        }
      }

      time_last_lidar = time_curr_lidar; //  update time
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  bool TryMAPInitialization()
  {
    //获取第一帧的平均加速度
    Eigen::Vector3d average_acc = -lidarFrameList->begin()->imuIntegrator.GetAverageAcc();
    //算出来是 -------average_acc=------------ 0.108004  0.0459886   -9.77121 有问题先屏蔽
    std::cout << "-------average_acc=------------" << average_acc << std::endl;
    //重力加速度与第一帧的加速度差
    //  average_acc = Eigen::Vector3d(0.0, 0.0, -9.805);   // 单位 m/s²

    double info_g = std::fabs(9.805 - average_acc.norm());
    //得到9.805加速度向量
    average_acc = average_acc * 9.805 / average_acc.norm();

    // calculate the initial gravity direction
    double para_quat[4];
    para_quat[0] = 1;
    para_quat[1] = 0;
    para_quat[2] = 0;
    para_quat[3] = 0;

    ceres::LocalParameterization *quatParam = new ceres::QuaternionParameterization();
    ceres::Problem problem_quat;

    problem_quat.AddParameterBlock(para_quat, 4, quatParam);

    problem_quat.AddResidualBlock(Cost_Initial_G::Create(average_acc),
                                  nullptr,
                                  para_quat);

    ceres::Solver::Options options_quat;
    ceres::Solver::Summary summary_quat;
    ceres::Solve(options_quat, &problem_quat, &summary_quat);

    Eigen::Quaterniond q_wg(para_quat[0], para_quat[1], para_quat[2], para_quat[3]);
    // std::cout << "-------q_wg=------------" << q_wg << std::endl;
    // std::cout << "-------lidarFrameList->size()=------------" << lidarFrameList->size() << std::endl;


    // build prior factor of LIO initialization
    Eigen::Vector3d prior_r = Eigen::Vector3d::Zero();
    Eigen::Vector3d prior_ba = Eigen::Vector3d::Zero();
    Eigen::Vector3d prior_bg = Eigen::Vector3d::Zero();
    std::vector<Eigen::Vector3d> prior_v;
    int v_size = lidarFrameList->size();
    for (int i = 0; i < v_size; i++)
    {
      prior_v.push_back(Eigen::Vector3d::Zero());
    }
    Sophus::SO3d SO3_R_wg(q_wg.toRotationMatrix());
    prior_r = SO3_R_wg.log();

    for (int i = 1; i < v_size; i++)
    {
      auto iter = lidarFrameList->begin();
      auto iter_next = lidarFrameList->begin();
      std::advance(iter, i - 1);
      std::advance(iter_next, i);

      Eigen::Vector3d velo_imu = (iter_next->P - iter->P) / (iter_next->timeStamp - iter->timeStamp);
      prior_v[i] = velo_imu;
    }
    prior_v[0] = prior_v[1];

    double para_v[v_size][3];
    double para_r[3];
    double para_ba[3];
    double para_bg[3];

    for (int i = 0; i < 3; i++)
    {
      para_r[i] = 0;
      para_ba[i] = 0;
      para_bg[i] = 0;
    }

    for (int i = 0; i < v_size; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        para_v[i][j] = prior_v[i][j];
      }
    }

    Eigen::Matrix<double, 3, 3> sqrt_information_r = 2000.0 * Eigen::Matrix<double, 3, 3>::Identity();
    Eigen::Matrix<double, 3, 3> sqrt_information_ba = 1000.0 * Eigen::Matrix<double, 3, 3>::Identity();
    Eigen::Matrix<double, 3, 3> sqrt_information_bg = 4000.0 * Eigen::Matrix<double, 3, 3>::Identity();
    Eigen::Matrix<double, 3, 3> sqrt_information_v = 4000.0 * Eigen::Matrix<double, 3, 3>::Identity();

    ceres::Problem::Options problem_options;
    ceres::Problem problem(problem_options);
    problem.AddParameterBlock(para_r, 3);
    problem.AddParameterBlock(para_ba, 3);
    problem.AddParameterBlock(para_bg, 3);
    for (int i = 0; i < v_size; i++)
    {
      problem.AddParameterBlock(para_v[i], 3);
    }

    // add CostFunction
    problem.AddResidualBlock(Cost_Initialization_Prior_R::Create(prior_r, sqrt_information_r),
                             nullptr,
                             para_r);

    problem.AddResidualBlock(Cost_Initialization_Prior_bv::Create(prior_ba, sqrt_information_ba),
                             nullptr,
                             para_ba);
    problem.AddResidualBlock(Cost_Initialization_Prior_bv::Create(prior_bg, sqrt_information_bg),
                             nullptr,
                             para_bg);

    for (int i = 0; i < v_size; i++)
    {
      problem.AddResidualBlock(Cost_Initialization_Prior_bv::Create(prior_v[i], sqrt_information_v),
                               nullptr,
                               para_v[i]);
    }

    for (int i = 1; i < v_size; i++)
    {
      auto iter = lidarFrameList->begin();
      auto iter_next = lidarFrameList->begin();
      std::advance(iter, i - 1);
      std::advance(iter_next, i);

      Eigen::Vector3d pi = iter->P;
      Sophus::SO3d SO3_Ri(iter->Q);
      Eigen::Vector3d ri = SO3_Ri.log();
      Eigen::Vector3d pj = iter_next->P;
      Sophus::SO3d SO3_Rj(iter_next->Q);
      Eigen::Vector3d rj = SO3_Rj.log();

      problem.AddResidualBlock(Cost_Initialization_IMU::Create(iter_next->imuIntegrator,
                                                               ri,
                                                               rj,
                                                               pj - pi,
                                                               Eigen::LLT<Eigen::Matrix<double, 9, 9>>(iter_next->imuIntegrator.GetCovariance().block<9, 9>(0, 0).inverse())
                                                                   .matrixL()
                                                                   .transpose()),
                               nullptr,
                               para_r,
                               para_v[i - 1],
                               para_v[i],
                               para_ba,
                               para_bg);
    }

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = false;
    options.linear_solver_type = ceres::DENSE_QR;
    options.num_threads = 6;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    Eigen::Vector3d r_wg(para_r[0], para_r[1], para_r[2]);
    GravityVector = Sophus::SO3d::exp(r_wg) * Eigen::Vector3d(0, 0, -9.805);

    Eigen::Vector3d ba_vec(para_ba[0], para_ba[1], para_ba[2]);
    Eigen::Vector3d bg_vec(para_bg[0], para_bg[1], para_bg[2]);

    std::cout << "after opt,ba:" << ba_vec.transpose() << ",bg: " << bg_vec.transpose()
              << ",g: " << GravityVector.transpose() << std::endl;

    if (ba_vec.norm() > 0.5 || bg_vec.norm() > 0.5)
    {
      ROS_WARN("Too Large Biases! Initialization Failed!");
      return false;
    }

    for (int i = 0; i < v_size; i++)
    {
      auto iter = lidarFrameList->begin();
      std::advance(iter, i);
      iter->ba = ba_vec;
      iter->bg = bg_vec;
      Eigen::Vector3d bv_vec(para_v[i][0], para_v[i][1], para_v[i][2]);
      if ((bv_vec - prior_v[i]).norm() > 2.0)
      {
        ROS_WARN("Too Large Velocity! Initialization Failed!");
        std::cout << "delta v norm: " << (bv_vec - prior_v[i]).norm() << std::endl;
        return false;
      }
      iter->V = bv_vec;
    }

    for (size_t i = 0; i < v_size - 1; i++)
    {
      auto laser_trans_i = lidarFrameList->begin();
      auto laser_trans_j = lidarFrameList->begin();
      std::advance(laser_trans_i, i);
      std::advance(laser_trans_j, i + 1);
      laser_trans_j->imuIntegrator.PreIntegration(laser_trans_i->timeStamp, laser_trans_i->bg, laser_trans_i->ba);
    }

    // //if IMU success initialized
    WINDOWSIZE = SLIDEWINDOWSIZE;
    while (lidarFrameList->size() > WINDOWSIZE)
    {
      lidarFrameList->pop_front();
    }
    Eigen::Vector3d Pwl = lidarFrameList->back().P;
    Eigen::Quaterniond Qwl = lidarFrameList->back().Q;
    lidarFrameList->back().P = Pwl;
    lidarFrameList->back().Q = Qwl;

    std::cout << "\n=============================================\n|         Initialization Successful         |"
              << "\n=============================================\n"
              << std::endl;

    
    // // 检查重力向量的合理性
    // double horizontal_component = sqrt(GravityVector(0)*GravityVector(0) + GravityVector(1)*GravityVector(1));
    // std::cout << "重力向量水平分量: " << horizontal_component << std::endl;
    // std::cout << "重力向量垂直分量: " << GravityVector(2) << std::endl;
    
    //       // 强制修正重力向量为垂直向下
    // std::cout << "=== 应用重力向量强制修正 ===" << std::endl;
    // std::cout << "原始重力向量: " << GravityVector.transpose() << std::endl;
    // std::cout << "外参旋转为单位矩阵，重力应垂直向下" << std::endl;
    
    // // 保持原始重力大小，但强制垂直向下
    // double gravity_magnitude = GravityVector.norm();
    // GravityVector = Eigen::Vector3d(0, 0, -gravity_magnitude);
    
    // std::cout << "修正后重力向量: " << GravityVector.transpose() << std::endl;
    // std::cout << "=================================" << std::endl;

    return true;
  }
  //去除点云畸变
  void RemoveLidarDistortion(CLOUD_PTR &laserCloud,
                             const Eigen::Matrix3d &dRlc, const Eigen::Vector3d &dtlc)
  {
    int PointsNum = laserCloud->points.size();
    for (int i = 0; i < PointsNum; i++)
    {
      Eigen::Vector3d startP;
      float s = laserCloud->points[i].normal_x; //  time intervel
      Eigen::Quaterniond qlc = Eigen::Quaterniond(dRlc).normalized();
      Eigen::Quaterniond delta_qlc = Eigen::Quaterniond::Identity().slerp(s, qlc).normalized(); // 插值
      const Eigen::Vector3d delta_Plc = s * dtlc;
      startP = delta_qlc * Eigen::Vector3d(laserCloud->points[i].x, laserCloud->points[i].y, laserCloud->points[i].z) + delta_Plc;
      Eigen::Vector3d _po = dRlc.transpose() * (startP - dtlc);

      laserCloud->points[i].x = _po(0);
      laserCloud->points[i].y = _po(1);
      laserCloud->points[i].z = _po(2);
      laserCloud->points[i].normal_x = 1.0;

      // if (std::fabs(kf.laserCloud->points[i].normal_z - 1.0) < 1e-5)
      //   kf.corner->push_back(kf.laserCloud->points[i]);
      // if (std::fabs(kf.laserCloud->points[i].normal_z - 2.0) < 1e-5)
      //   kf.surf->push_back(kf.laserCloud->points[i]);
    }
    /*std::cout << "bef-surf: " << kf.surf->size() << ",corner: " << kf.corner->size() << std::endl;
    ds_surf_.setInputCloud(kf.surf);
    ds_surf_.filter(*kf.surf);
    ds_corner_.setInputCloud(kf.corner);
    ds_corner_.filter(*kf.corner);
    std::cout << "aft-surf: " << kf.surf->size() << ",corner: " << kf.corner->size() << std::endl;*/
  }
  // 找到两帧之间的imu数据，
  bool fetchImuMsgs(double startTime, double endTime, std::vector<sensor_msgs::ImuConstPtr> &vimuMsg)
  {
    //startTime：上一帧 endTime：这一帧
    // std::cout << "上一帧时间戳=" << startTime << "这一帧时间戳=" << endTime << std::endl;
    double current_time = 0;
    vimuMsg.clear();
    while (true)
    {
      if (_imuMsgQueue.empty())
        break;
      if (_imuMsgQueue.back()->header.stamp.toSec() < endTime ||
          _imuMsgQueue.front()->header.stamp.toSec() >= endTime)
        break;
      sensor_msgs::ImuConstPtr tmpimumsg = _imuMsgQueue.front();
      double time = tmpimumsg->header.stamp.toSec();
      if (time <= endTime && time > startTime)
      {
        vimuMsg.push_back(tmpimumsg);
        current_time = time;
        _imuMsgQueue.pop_front();
        if (time == endTime)
          break;
      }
      else
      {
        if (time <= startTime)
        {
          _imuMsgQueue.pop_front();
        }
        else
        {
          double dt_1 = endTime - current_time;
          double dt_2 = time - endTime;
          ROS_ASSERT(dt_1 >= 0);
          ROS_ASSERT(dt_2 >= 0);
          ROS_ASSERT(dt_1 + dt_2 > 0);
          double w1 = dt_2 / (dt_1 + dt_2);
          double w2 = dt_1 / (dt_1 + dt_2);
          sensor_msgs::ImuPtr theLastIMU(new sensor_msgs::Imu);
          theLastIMU->linear_acceleration.x = w1 * vimuMsg.back()->linear_acceleration.x + w2 * tmpimumsg->linear_acceleration.x;
          theLastIMU->linear_acceleration.y = w1 * vimuMsg.back()->linear_acceleration.y + w2 * tmpimumsg->linear_acceleration.y;
          theLastIMU->linear_acceleration.z = w1 * vimuMsg.back()->linear_acceleration.z + w2 * tmpimumsg->linear_acceleration.z;
          theLastIMU->angular_velocity.x = w1 * vimuMsg.back()->angular_velocity.x + w2 * tmpimumsg->angular_velocity.x;
          theLastIMU->angular_velocity.y = w1 * vimuMsg.back()->angular_velocity.y + w2 * tmpimumsg->angular_velocity.y;
          theLastIMU->angular_velocity.z = w1 * vimuMsg.back()->angular_velocity.z + w2 * tmpimumsg->angular_velocity.z;
          theLastIMU->header.stamp.fromSec(endTime);
          vimuMsg.emplace_back(theLastIMU);
          break;
        }
      }
    }
    return !vimuMsg.empty();
  }
  //找到对应点线特征，填入edge vLineFeatures
  void processPointToLine(std::vector<ceres::CostFunction *> &edges,
                          std::vector<FeatureLine> &vLineFeatures,
                          const pcl::PointCloud<PointType>::Ptr &laserCloudCorner,
                          const pcl::PointCloud<PointType>::Ptr &laserCloudCornerGlobal,
                          const pcl::KdTreeFLANN<PointType>::Ptr &kdtreeGlobal,
                          const pcl::PointCloud<PointType>::Ptr &laserCloudCornerLocal,
                          const pcl::KdTreeFLANN<PointType>::Ptr &kdtreeLocal,
                          const Eigen::Matrix4d &exTlb,
                          const Eigen::Matrix4d &m4d)
  {
    std::lock_guard<std::mutex> lock(line_feature_mutex_);

    if (!laserCloudCorner || laserCloudCorner->empty()) {
        std::cout << "ERROR: laserCloudCorner is empty!" << std::endl;
        return;
    }
    
    if ((!laserCloudCornerGlobal || laserCloudCornerGlobal->empty()) && 
        (!laserCloudCornerLocal || laserCloudCornerLocal->empty())) {
        std::cout << "ERROR: Both global and local corner maps are empty!" << std::endl;
        return;
    }

    // std::cout << ">>> 进入 processPointToLine" << std::endl;

    Eigen::Matrix4d Tbl = Eigen::Matrix4d::Identity();
    Tbl.topLeftCorner(3, 3) = exTlb.topLeftCorner(3, 3).transpose();
    Tbl.topRightCorner(3, 1) = -1.0 * Tbl.topLeftCorner(3, 3) * exTlb.topRightCorner(3, 1);
    if (!vLineFeatures.empty())
    {
      for (const auto &l : vLineFeatures)
      {
        auto *e = Cost_NavState_IMU_Line::Create(l.pointOri,
                                                 l.lineP1,
                                                 l.lineP2,
                                                 Tbl,
                                                 Eigen::Matrix<double, 1, 1>(1 / IMUIntegrator::lidar_m));
        edges.push_back(e);
      }
      return;
    }
    PointType _pointOri, _pointSel, _coeff;
    std::vector<int> _pointSearchInd;
    std::vector<float> _pointSearchSqDis;
    std::vector<int> _pointSearchInd2;
    std::vector<float> _pointSearchSqDis2;

    Eigen::Matrix<double, 3, 3> _matA1;
    _matA1.setZero();

    int laserCloudCornerStackNum = laserCloudCorner->points.size();
    pcl::PointCloud<PointType>::Ptr kd_pointcloud(new pcl::PointCloud<PointType>);
    int debug_num1 = 0;
    int debug_num2 = 0;
    int debug_num12 = 0;
    int debug_num22 = 0;

    for (int i = 0; i < laserCloudCornerStackNum; i++)
    {
      _pointOri = laserCloudCorner->points[i];
      MAP_MANAGER::pointAssociateToMap(&_pointOri, &_pointSel, m4d);

      //  for global
      if (laserCloudCornerGlobal->points.size() > 100)
      {
        kdtreeGlobal->nearestKSearch(_pointSel, 5, _pointSearchInd2, _pointSearchSqDis2);
        if (_pointSearchSqDis2[4] < thres_dist)
        {

          debug_num2++;
          float cx = 0;
          float cy = 0;
          float cz = 0;
          for (int j = 0; j < 5; j++)
          {
            cx += laserCloudCornerGlobal->points[_pointSearchInd2[j]].x;
            cy += laserCloudCornerGlobal->points[_pointSearchInd2[j]].y;
            cz += laserCloudCornerGlobal->points[_pointSearchInd2[j]].z;
          }
          cx /= 5;
          cy /= 5;
          cz /= 5;

          float a11 = 0;
          float a12 = 0;
          float a13 = 0;
          float a22 = 0;
          float a23 = 0;
          float a33 = 0;
          for (int j = 0; j < 5; j++)
          {
            float ax = laserCloudCornerGlobal->points[_pointSearchInd2[j]].x - cx;
            float ay = laserCloudCornerGlobal->points[_pointSearchInd2[j]].y - cy;
            float az = laserCloudCornerGlobal->points[_pointSearchInd2[j]].z - cz;

            a11 += ax * ax;
            a12 += ax * ay;
            a13 += ax * az;
            a22 += ay * ay;
            a23 += ay * az;
            a33 += az * az;
          }
          a11 /= 5;
          a12 /= 5;
          a13 /= 5;
          a22 /= 5;
          a23 /= 5;
          a33 /= 5;

          _matA1(0, 0) = a11;
          _matA1(0, 1) = a12;
          _matA1(0, 2) = a13;
          _matA1(1, 0) = a12;
          _matA1(1, 1) = a22;
          _matA1(1, 2) = a23;
          _matA1(2, 0) = a13;
          _matA1(2, 1) = a23;
          _matA1(2, 2) = a33;



          Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(_matA1);
          Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);

          if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
          {
            debug_num22++;
            float x1 = cx + 0.1 * unit_direction[0];
            float y1 = cy + 0.1 * unit_direction[1];
            float z1 = cz + 0.1 * unit_direction[2];
            float x2 = cx - 0.1 * unit_direction[0];
            float y2 = cy - 0.1 * unit_direction[1];
            float z2 = cz - 0.1 * unit_direction[2];

            Eigen::Vector3d tripod1(x1, y1, z1);
            Eigen::Vector3d tripod2(x2, y2, z2);

            auto *e = Cost_NavState_IMU_Line::Create(Eigen::Vector3d (_pointOri.x, _pointOri.y, _pointOri.z),
                                                    tripod1,
                                                    tripod2,
                                                    Tbl,
                                                    Eigen::Matrix<double, 1, 1>(1 / IMUIntegrator::lidar_m));                                    
            edges.push_back(e);

            vLineFeatures.emplace_back(Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                                       tripod1,
                                       tripod2);

            vLineFeatures.back().ComputeError(m4d);
          }
        }
      }

      //  for local
      if (laserCloudCornerLocal->points.size() > 20)
      {
        kdtreeLocal->nearestKSearch(_pointSel, 5, _pointSearchInd2, _pointSearchSqDis2);
        if (_pointSearchSqDis2[4] < thres_dist)
        {

          debug_num2++;
          float cx = 0;
          float cy = 0;
          float cz = 0;
          for (int j = 0; j < 5; j++)
          {
            cx += laserCloudCornerLocal->points[_pointSearchInd2[j]].x;
            cy += laserCloudCornerLocal->points[_pointSearchInd2[j]].y;
            cz += laserCloudCornerLocal->points[_pointSearchInd2[j]].z;
          }
          cx /= 5;
          cy /= 5;
          cz /= 5;

          float a11 = 0;
          float a12 = 0;
          float a13 = 0;
          float a22 = 0;
          float a23 = 0;
          float a33 = 0;
          for (int j = 0; j < 5; j++)
          {
            float ax = laserCloudCornerLocal->points[_pointSearchInd2[j]].x - cx;
            float ay = laserCloudCornerLocal->points[_pointSearchInd2[j]].y - cy;
            float az = laserCloudCornerLocal->points[_pointSearchInd2[j]].z - cz;

            a11 += ax * ax;
            a12 += ax * ay;
            a13 += ax * az;
            a22 += ay * ay;
            a23 += ay * az;
            a33 += az * az;
          }
          a11 /= 5;
          a12 /= 5;
          a13 /= 5;
          a22 /= 5;
          a23 /= 5;
          a33 /= 5;

          _matA1(0, 0) = a11;
          _matA1(0, 1) = a12;
          _matA1(0, 2) = a13;
          _matA1(1, 0) = a12;
          _matA1(1, 1) = a22;
          _matA1(1, 2) = a23;
          _matA1(2, 0) = a13;
          _matA1(2, 1) = a23;
          _matA1(2, 2) = a33;

          Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(_matA1);
          Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);

          if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
          {
            debug_num22++;
            float x1 = cx + 0.1 * unit_direction[0];
            float y1 = cy + 0.1 * unit_direction[1];
            float z1 = cz + 0.1 * unit_direction[2];
            float x2 = cx - 0.1 * unit_direction[0];
            float y2 = cy - 0.1 * unit_direction[1];
            float z2 = cz - 0.1 * unit_direction[2];

            Eigen::Vector3d tripod1(x1, y1, z1);
            Eigen::Vector3d tripod2(x2, y2, z2);
            auto *e = Cost_NavState_IMU_Line::Create(Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                                                     tripod1,
                                                     tripod2,
                                                     Tbl,
                                                     Eigen::Matrix<double, 1, 1>(1 / IMUIntegrator::lidar_m));
            edges.push_back(e);
            vLineFeatures.emplace_back(Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                                       tripod1,
                                       tripod2);
            vLineFeatures.back().ComputeError(m4d);
          }
        }
      }
    }
  }
  //找到对应点面特征，填入edge vPlanFeatures
  void processPointToPlanVec(std::vector<ceres::CostFunction *> &edges,
                             std::vector<FeaturePlanVec> &vPlanFeatures,
                             const pcl::PointCloud<PointType>::Ptr &laserCloudSurf,
                             const pcl::PointCloud<PointType>::Ptr &laserCloudSurfGlobal,
                             const pcl::KdTreeFLANN<PointType>::Ptr &kdtreeGlobal,
                             const pcl::PointCloud<PointType>::Ptr &laserCloudSurfLocal,
                             const pcl::KdTreeFLANN<PointType>::Ptr &kdtreeLocal,
                             const Eigen::Matrix4d &exTlb,
                             const Eigen::Matrix4d &m4d)
  {
    std::lock_guard<std::mutex> lock(plan_feature_mutex_);

    Eigen::Matrix4d Tbl = Eigen::Matrix4d::Identity();
    Tbl.topLeftCorner(3, 3) = exTlb.topLeftCorner(3, 3).transpose();
    Tbl.topRightCorner(3, 1) = -1.0 * Tbl.topLeftCorner(3, 3) * exTlb.topRightCorner(3, 1);
    if (!vPlanFeatures.empty())
    {
      for (const auto &p : vPlanFeatures)
      {
        auto *e = Cost_NavState_IMU_Plan_Vec::Create(p.pointOri,
                                                     p.pointProj,
                                                     Tbl,
                                                     p.sqrt_info);
        edges.push_back(e);
      }
      return;
    }
    PointType _pointOri, _pointSel, _coeff;
    std::vector<int> _pointSearchInd;
    std::vector<float> _pointSearchSqDis;
    std::vector<int> _pointSearchInd2;
    std::vector<float> _pointSearchSqDis2;

    Eigen::Matrix<double, 5, 3> _matA0;
    _matA0.setZero();
    Eigen::Matrix<double, 5, 1> _matB0;
    _matB0.setOnes();
    _matB0 *= -1;
    Eigen::Matrix<double, 3, 1> _matX0;
    _matX0.setZero();
    int laserCloudSurfStackNum = laserCloudSurf->points.size();

    int debug_num1 = 0;
    int debug_num2 = 0;
    int debug_num12 = 0;
    int debug_num22 = 0;

    int out_flag = 0;
    for (int i = 0; i < laserCloudSurfStackNum; i++)
    {
      _pointOri = laserCloudSurf->points[i];
      MAP_MANAGER::pointAssociateToMap(&_pointOri, &_pointSel, m4d);
      //  for global
      if (laserCloudSurfGlobal->points.size() > 200)
      {
        kdtreeGlobal->nearestKSearch(_pointSel, 5, _pointSearchInd2, _pointSearchSqDis2);
        if (_pointSearchSqDis2[4] < thres_dist)
        {
          out_flag ++ ;
          debug_num2++;
          for (int j = 0; j < 5; j++)
          {
            _matA0(j, 0) = laserCloudSurfGlobal->points[_pointSearchInd2[j]].x;
            _matA0(j, 1) = laserCloudSurfGlobal->points[_pointSearchInd2[j]].y;
            _matA0(j, 2) = laserCloudSurfGlobal->points[_pointSearchInd2[j]].z;
          }
          _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);

          float pa = _matX0(0, 0);
          float pb = _matX0(1, 0);
          float pc = _matX0(2, 0);
          float pd = 1;

          float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
          pa /= ps;
          pb /= ps;
          pc /= ps;
          pd /= ps;

          bool planeValid = true;
          for (int j = 0; j < 5; j++)
          {
            if (std::fabs(pa * laserCloudSurfGlobal->points[_pointSearchInd2[j]].x +
                          pb * laserCloudSurfGlobal->points[_pointSearchInd2[j]].y +
                          pc * laserCloudSurfGlobal->points[_pointSearchInd2[j]].z + pd) > 0.2)
            {
              planeValid = false;
              break;
            }
          }

          if (planeValid)
          {
            debug_num22++;
            double dist = pa * _pointSel.x +
                          pb * _pointSel.y +
                          pc * _pointSel.z + pd;
            Eigen::Vector3d omega(pa, pb, pc);
            Eigen::Vector3d point_proj = Eigen::Vector3d(_pointSel.x, _pointSel.y, _pointSel.z) - (dist * omega);
            Eigen::Vector3d e1(1, 0, 0);
            Eigen::Matrix3d J = e1 * omega.transpose();
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(J, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d R_svd = svd.matrixV() * svd.matrixU().transpose();
            Eigen::Matrix3d info = (1.0 / IMUIntegrator::lidar_m) * Eigen::Matrix3d::Identity();
            info(1, 1) *= plan_weight_tan;
            info(2, 2) *= plan_weight_tan;
            Eigen::Matrix3d sqrt_info = info * R_svd.transpose();

            auto *e = Cost_NavState_IMU_Plan_Vec::Create(Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                                                         point_proj,
                                                         Tbl,
                                                         sqrt_info);
            edges.push_back(e);
            vPlanFeatures.emplace_back(Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                                       point_proj,
                                       sqrt_info);
            vPlanFeatures.back().ComputeError(m4d);
          }
        }
      }

      if (laserCloudSurfLocal->points.size() > 20)
      {
        kdtreeLocal->nearestKSearch(_pointSel, 5, _pointSearchInd2, _pointSearchSqDis2);
        if (_pointSearchSqDis2[4] < thres_dist)
        {
          debug_num2++;
          for (int j = 0; j < 5; j++)
          {
            _matA0(j, 0) = laserCloudSurfLocal->points[_pointSearchInd2[j]].x;
            _matA0(j, 1) = laserCloudSurfLocal->points[_pointSearchInd2[j]].y;
            _matA0(j, 2) = laserCloudSurfLocal->points[_pointSearchInd2[j]].z;
          }
          _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);

          float pa = _matX0(0, 0);
          float pb = _matX0(1, 0);
          float pc = _matX0(2, 0);
          float pd = 1;

          float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
          pa /= ps;
          pb /= ps;
          pc /= ps;
          pd /= ps;

          bool planeValid = true;
          for (int j = 0; j < 5; j++)
          {
            if (std::fabs(pa * laserCloudSurfLocal->points[_pointSearchInd2[j]].x +
                          pb * laserCloudSurfLocal->points[_pointSearchInd2[j]].y +
                          pc * laserCloudSurfLocal->points[_pointSearchInd2[j]].z + pd) > 0.2)
            {
              planeValid = false;
              break;
            }
          }

          if (planeValid)
          {
            debug_num22++;
            double dist = pa * _pointSel.x +
                          pb * _pointSel.y +
                          pc * _pointSel.z + pd;
            Eigen::Vector3d omega(pa, pb, pc);
            Eigen::Vector3d point_proj = Eigen::Vector3d(_pointSel.x, _pointSel.y, _pointSel.z) - (dist * omega);
            Eigen::Vector3d e1(1, 0, 0);
            Eigen::Matrix3d J = e1 * omega.transpose();
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(J, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3d R_svd = svd.matrixV() * svd.matrixU().transpose();
            Eigen::Matrix3d info = (1.0 / IMUIntegrator::lidar_m) * Eigen::Matrix3d::Identity();
            info(1, 1) *= plan_weight_tan;
            info(2, 2) *= plan_weight_tan;
            Eigen::Matrix3d sqrt_info = info * R_svd.transpose();

            auto *e = Cost_NavState_IMU_Plan_Vec::Create(Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                                                         point_proj,
                                                         Tbl,
                                                         sqrt_info);
            edges.push_back(e);
            vPlanFeatures.emplace_back(Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                                       point_proj,
                                       sqrt_info);
            vPlanFeatures.back().ComputeError(m4d);
          }
        }
      }
    }

    if (out_flag < laserCloudSurfStackNum * 0.6 ){
      std::cout << "map outside!" << std::endl;
    }

    out_flag = 0;
  }

  void MapIncrementLocal(LidarFrame &kframe)
  {
    int laserCloudCornerStackNum = kframe.corner->points.size();
    int laserCloudSurfStackNum = kframe.surf->points.size();
    Eigen::Matrix4d pose_in_map = Eigen::Matrix4d::Identity();
    pose_in_map.topLeftCorner(3, 3) = kframe.Q.toRotationMatrix();
    pose_in_map.topRightCorner(3, 1) = kframe.P;
    PointType pointSel;
    PointType pointSel2;
    size_t Id = localMapID % localMapWindowSize;
    localCornerMap[Id]->clear();
    localSurfMap[Id]->clear();
    for (int i = 0; i < laserCloudCornerStackNum; i++)
    {
      MAP_MANAGER::pointAssociateToMap(&kframe.corner->points[i], &pointSel, pose_in_map);
      localCornerMap[Id]->push_back(pointSel);
    }
    for (int i = 0; i < laserCloudSurfStackNum; i++)
    {
      MAP_MANAGER::pointAssociateToMap(&kframe.surf->points[i], &pointSel2, pose_in_map);
      localSurfMap[Id]->push_back(pointSel2);
    }

    for (int i = 0; i < localMapWindowSize; i++)
    {
      *laserCloudCornerFromLocal += *localCornerMap[i];
      *laserCloudSurfFromLocal += *localSurfMap[i];
    }
    pcl::PointCloud<PointType>::Ptr temp(new pcl::PointCloud<PointType>());
    ds_corner_.setInputCloud(laserCloudCornerFromLocal);
    ds_corner_.filter(*temp);
    laserCloudCornerFromLocal = temp;
    pcl::PointCloud<PointType>::Ptr temp2(new pcl::PointCloud<PointType>());
    ds_surf_.setInputCloud(laserCloudSurfFromLocal);
    ds_surf_.filter(*temp2);
    laserCloudSurfFromLocal = temp2;

    localMapID++;
  }

  bool ICPScanMatchGlobal(std::list<LidarFrame> &kframeList)
    {
    if (kframeList.size() != 1)
      std::cout << "may error,only process one lidar frame" << std::endl;

    auto &kframe = kframeList.front();
    CLOUD_PTR surf(new CLOUD());
    for (const auto &p : kframe.laserCloud->points)
    {
      // if (std::fabs(p.normal_z - 2.0) < 1e-5)
        surf->push_back(p);
    }
    ds_surf_.setInputCloud(surf);
    ds_surf_.filter(*surf);

    TicToc tc;
    tc.tic();
    CLOUD_PTR cloud_icp(new CLOUD());
    // *cloud_icp += *TransformPointCloud(corner, &initpose);
    
    *cloud_icp += *TransformPointCloud(surf, &initpose);
    //  std::cout << "initpose.x used" << initpose.x <<std::endl;
    //  std::cout << "initpose.y used" << initpose.y <<std::endl;
    //  std::cout << "initpose.z used" << initpose.z <<std::endl;

    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(50); //配对点最大距离
    icp.setMaximumIterations(30);   //最大迭代次数
    icp.setTransformationEpsilon(1e-4);  //步长阈值
    icp.setEuclideanFitnessEpsilon(1e-4);  //结束迭代的误差阈值
    icp.setRANSACIterations(0);  //禁用RANSAC,动态环境使用

    icp.setInputSource(cloud_icp);
    icp.setInputTarget(surround_surf);
    CLOUD_PTR unused_result(new CLOUD());
    icp.align(*unused_result);

    if (icp.hasConverged() == false || icp.getFitnessScore() > 0.4)
    {
      std::cout << ANSI_COLOR_RED << "initial loc failed...,score: " << icp.getFitnessScore() << ANSI_COLOR_RESET << std::endl;
      return false;
    }
    Eigen::Affine3f correct_transform;
    correct_transform = icp.getFinalTransformation();
    Eigen::Matrix4d curr_pose = toMatrix(initpose);

    Eigen::Matrix4d pose = correct_transform.matrix().cast<double>() * curr_pose;

    kframe.Q = pose.block<3, 3>(0, 0);
    kframe.P = pose.topRightCorner(3, 1); //  update pose here

    double tt = tc.toc();
    std::cout << "icp takes: " << tt << "ms" << std::endl;
    CLOUD_PTR output(new CLOUD);

    pcl::transformPointCloud(*surf, *output, pose);
    sensor_msgs::PointCloud2 msg_target;
    // pcl::toROSMsg(*cloud_icp, msg_target);
    pcl::toROSMsg(*output, msg_target);
    msg_target.header.stamp = ros::Time::now();
    msg_target.header.frame_id = "world";
    pub_surf_.publish(msg_target);

    return true;
  }

  Eigen::Matrix4d toMatrix(PointTypePose &p)
  {
    Eigen::Matrix4d odom = Eigen::Matrix4d::Identity();
    Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(p.roll, Eigen::Vector3d::UnitX()));
    Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(p.pitch, Eigen::Vector3d::UnitY()));
    Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(p.yaw, Eigen::Vector3d::UnitZ()));
    Eigen::Quaterniond rotation = yawAngle * pitchAngle * rollAngle;
    odom.block(0, 0, 3, 3) = rotation.toRotationMatrix();
    odom(0, 3) = p.x, odom(1, 3) = p.y, odom(2, 3) = p.z;
    return odom;
  }

  CLOUD_PTR TransformPointCloud(CLOUD_PTR cloudIn, PointTypePose *transformIn)
  {
    CLOUD_PTR cloudOut(new CLOUD());
    PointType *pointfrom;
    PointType pointTo;

    int cloudSize = cloudIn->points.size();
    cloudOut->resize(cloudSize);
    for (int i = 0; i < cloudSize; i++)
    {
      pointfrom = &cloudIn->points[i];
      float x1 = pointfrom->x;
      float y1 = cos(transformIn->roll) * pointfrom->y - sin(transformIn->roll) * pointfrom->z;
      float z1 = sin(transformIn->roll) * pointfrom->y + cos(transformIn->roll) * pointfrom->z;

      float x2 = cos(transformIn->pitch) * x1 + sin(transformIn->pitch) * z1;
      float y2 = y1;
      float z2 = -sin(transformIn->pitch) * x1 + cos(transformIn->pitch) * z1;

      pointTo.x = cos(transformIn->yaw) * x2 - sin(transformIn->yaw) * y2 + transformIn->x;
      pointTo.y = sin(transformIn->yaw) * x2 + cos(transformIn->yaw) * y2 + transformIn->y;
      pointTo.z = z2 + transformIn->z;
      pointTo.intensity = pointfrom->intensity;

      cloudOut->points[i] = pointTo;
    }
    return cloudOut;
  }

  bool extractSurroundKeyFrames(const PointType &p)
  {
    TicToc tc;
    tc.tic();
    std::cout << "-----extract surround keyframes ------ " << std::endl;
    std::vector<int> point_search_idx_;
    std::vector<float> point_search_dist_;
    kdtree_keyposes_3d_->radiusSearch(p, surround_search_radius_, point_search_idx_, point_search_dist_, 0);
    surround_surf->clear();
    surround_corner->clear();
    std::cout << "point_search_idx_.size() = " << point_search_idx_.size() << std::endl;
    std::cout << "map.surf_keyframes_.size() = " << map.surf_keyframes_.size() << std::endl;
    for (int i = 0; i < point_search_idx_.size(); ++i)
    {
      std::cout << "point_search_idx_ = " << point_search_idx_[i] << std::endl;
      // *surround_surf += *map.surf_keyframes_[i];
      // *surround_corner += *map.corner_keyframes_[i];
      *surround_surf += *map.surf_keyframes_[point_search_idx_[i]];
      *surround_corner += *map.corner_keyframes_[point_search_idx_[i]];
    }

    ds_corner_.setInputCloud(surround_corner);
    ds_corner_.filter(*surround_corner);
    ds_surf_.setInputCloud(surround_surf);
    ds_surf_.filter(*surround_surf);
    //新增
    // laserCloudCornerFromLocal = surround_corner;
    // laserCloudSurfFromLocal = surround_surf;

        // ========== 新增：发布周围点云 ==========
    // 发布周围角点
    sensor_msgs::PointCloud2 msg_surround_corner;
    pcl::toROSMsg(*surround_corner, msg_surround_corner);
    msg_surround_corner.header.stamp = ros::Time::now();
    msg_surround_corner.header.frame_id = "world";
    pub_surround_corner_.publish(msg_surround_corner);
    
    // 发布周围面点  
    sensor_msgs::PointCloud2 msg_surround_surf;
    pcl::toROSMsg(*surround_surf, msg_surround_surf);
    msg_surround_surf.header.stamp = ros::Time::now();
    msg_surround_surf.header.frame_id = "world";
    pub_surround_surf_.publish(msg_surround_surf);
    
    std::cout << "Published init pose surround corner: " << surround_corner->size() 
              << " points, init pose surround surf: " << surround_surf->size() << " points" << std::endl;
    // ========== 新增结束 ==========
    

    double tt = tc.toc();
    std::cout << __FUNCTION__ << ",takes: " << tt << "ms" << std::endl;

    return true;
  }

  bool loadmap()
  {
    std::cout << ANSI_COLOR_YELLOW << "file dir: " << filename << ANSI_COLOR_RESET << std::endl;
    CLOUD_PTR globalCornerCloud(new CLOUD);
    CLOUD_PTR globalSurfCloud(new CLOUD);

    std::string fn_poses_ = filename + "/trajectory.pcd";
    std::string fn_corner_ = filename + "/CornerMap.pcd";
    std::string fn_surf_ = filename + "/SurfMap.pcd";
    std::string fn_global_ = filename + "/GlobalMap.pcd";

    if (pcl::io::loadPCDFile(fn_poses_, *map.cloudKeyPoses3D_) == -1 ||
        pcl::io::loadPCDFile(fn_corner_, *globalCornerCloud) == -1 ||
        pcl::io::loadPCDFile(fn_surf_, *globalSurfCloud) == -1 ||
        pcl::io::loadPCDFile(fn_global_, *map.globalMapCloud_))
    {
      std::cout << ANSI_COLOR_RED << "couldn't load pcd file" << ANSI_COLOR_RESET << std::endl;
      return false;
    }

    map.corner_keyframes_.resize(map.cloudKeyPoses3D_->points.size());
    map.surf_keyframes_.resize(map.cloudKeyPoses3D_->points.size());
    //根据关键帧数量,初始化角点面点关键帧容器
    for (int i = 0; i < map.cloudKeyPoses3D_->points.size(); ++i)
    {
      map.corner_keyframes_[i] = CLOUD_PTR(new CLOUD);
      map.surf_keyframes_[i] = CLOUD_PTR(new CLOUD);
    }
    //要求保存的地图里,点的intensity字段保存的是其所在的关键帧
    for (int i = 0; i < globalCornerCloud->points.size(); ++i)
    {
      const auto &p = globalCornerCloud->points[i];
      map.corner_keyframes_[int(p.intensity)]->points.push_back(p);
      // std::cout << "globalCornerCloud.intensity=" << int(p.intensity) << std::endl;
    }

    for (int i = 0; i < globalSurfCloud->points.size(); ++i)
    {
      const auto &p = globalSurfCloud->points[i];
      map.surf_keyframes_[int(p.intensity)]->points.push_back(p);
      // std::cout << "globalSurfCloud.intensity=" << int(p.intensity) << std::endl;

    }

    ds_corner_.setInputCloud(globalCornerCloud);
    ds_corner_.filter(*map.globalCornerMapCloud_);
    std::cout << "全局角点地图降采样前后 " <<globalCornerCloud->size() << ", " << map.globalCornerMapCloud_->size() << std::endl;
    ds_surf_.setInputCloud(globalSurfCloud);
    ds_surf_.filter(*map.globalSurfMapCloud_);
    std::cout << "全局面点地图降采样前后 " <<globalSurfCloud->size() << ", " << map.globalSurfMapCloud_->size() << std::endl;
    return true;
  }

  void saveTrajectoryTUMformat(std::fstream &fout, std::string &stamp, Eigen::Vector3d &xyz, Eigen::Quaterniond &xyzw)
  {
    fout << stamp << " " << xyz[0] << " " << xyz[1] << " " << xyz[2] << " " << xyzw.x() << " " << xyzw.y() << " " << xyzw.z() << " " << xyzw.w() << std::endl;
  }

  void saveTrajectoryTUMformat(std::fstream &fout, std::string &stamp, double x, double y, double z, double qx, double qy, double qz, double qw)
  {
    fout << stamp << " " << x << " " << y << " " << z << " " << qx << " " << qy << " " << qz << " " << qw << std::endl;
  }

  bool checkPointCloudStatus(const pcl::PointCloud<PointType>::Ptr& cloud)
  {
      if (!cloud)
          return false;
      else if (cloud->empty())
          return false;
      else
          return true ;
  }

  bool loadQRCodePosesFromFile(const std::string& file_path) {
      qrcode_map_poses_.clear();
          
      std::ifstream file(file_path);
      if (!file.is_open()) {
        std::cerr << "Error: Cannot open QR code pose file: " << file_path << std::endl;
        return false;
      }
          
      std::string line;
      int line_num = 0;
      int valid_poses = 0;
      
      std::cout << "Loading QR code poses from: " << file_path << std::endl;
      
      while (std::getline(file, line)) {
          line_num++;
          
          // 跳过注释行和空行
          if (isCommentOrEmpty(line)) {
              continue;
          }
          
          // 打印原始行用于调试
          std::cout << "Processing line " << line_num << ": " << line << std::endl;
          
          QRCodeMapPose pose;
          if (parseQRCodeLine(line, pose)) {
              // 检查tag_id是否重复
              if (qrcode_map_poses_.find(pose.tag_id) != qrcode_map_poses_.end()) {
                  std::cerr << "Warning: Duplicate tag_id " << pose.tag_id 
                            << " found at line " << line_num << ". Overwriting." << std::endl;
              }
              
              qrcode_map_poses_[pose.tag_id] = pose;
              valid_poses++;
              std::cout << "Successfully loaded QR code pose for tag " << pose.tag_id << std::endl;
          } else {
              std::cerr << "Warning: Failed to parse line " << line_num 
                        << ": " << line << std::endl;
              // 尝试手动解析，看看问题出在哪里
              std::cout << "Manual debug of line " << line_num << ":" << std::endl;
              std::cout << "  Raw line: '" << line << "'" << std::endl;
              std::cout << "  Line length: " << line.length() << std::endl;
              
              // 显示每个字符的ASCII值
              for (size_t i = 0; i < line.length(); ++i) {
                  std::cout << "  char[" << i << "]: '" << line[i] 
                            << "' (ASCII: " << static_cast<int>(line[i]) << ")" << std::endl;
              }
          }
      }
      
      file.close();
      
      if (valid_poses > 0) {
          std::cout << "Successfully loaded " << valid_poses 
                    << " QR code poses from " << file_path << std::endl;
          
          // 打印所有加载的二维码信息
          std::cout << "\nLoaded QR codes:" << std::endl;
          for (const auto& pair : qrcode_map_poses_) {
              std::cout << "  Tag " << pair.first << ": "
                        << "pos=[" << pair.second.position.x() << ", " 
                        << pair.second.position.y() << ", " << pair.second.position.z() << "] "
                        << "quat=[" << pair.second.orientation.x() << ", "
                        << pair.second.orientation.y() << ", " << pair.second.orientation.z() << ", "
                        << pair.second.orientation.w() << "]" << std::endl;
          }
          
          return true;
      } else {
          std::cerr << "Error: No valid QR code poses loaded from " << file_path << std::endl;
          return false;
      }
  }

  bool parseQRCodeLine(const std::string& line, QRCodeMapPose& qr_pose) {
      // 去除行首行尾的空格
      std::string trimmed_line = line;
      
      // 删除行首空格
      trimmed_line.erase(0, trimmed_line.find_first_not_of(" \t"));
      // 删除行尾空格
      trimmed_line.erase(trimmed_line.find_last_not_of(" \t") + 1);
      
      // 如果行已经为空，返回false
      if (trimmed_line.empty()) {
          return false;
      }
      
      std::istringstream iss(trimmed_line);
      
      int tag_id;
      double x, y, z;
      double qx, qy, qz, qw;
      
      // 尝试读取8个数值
      if (iss >> tag_id >> x >> y >> z >> qx >> qy >> qz >> qw) {
          // 验证四元数是否有效
          double norm_squared = qx*qx + qy*qy + qz*qz + qw*qw;
          if (norm_squared < 0.1 || norm_squared > 10.0) {
              std::cerr << "Warning: Invalid quaternion norm for tag " << tag_id 
                        << ": " << norm_squared << ". Using identity quaternion." << std::endl;
              // 使用单位四元数
              qr_pose = QRCodeMapPose(tag_id, x, y, z, 0.0, 0.0, 0.0, 1.0);
          } else {
              // 归一化四元数
              double norm = sqrt(norm_squared);
              qx /= norm;
              qy /= norm;
              qz /= norm;
              qw /= norm;
              qr_pose = QRCodeMapPose(tag_id, x, y, z, qx, qy, qz, qw);
          }
          
          std::cout << "Parsed QR code " << tag_id 
                    << ": pos=[" << x << ", " << y << ", " << z << "] "
                    << "quat=[" << qx << ", " << qy << ", " << qz << ", " << qw << "]" << std::endl;
          return true;
      }
    
      std::cerr << "Failed to parse line: " << line << std::endl;
      return false;
  }

  bool isCommentOrEmpty(const std::string& line) {
      // 检查是否为空行
      if (line.empty()) {
          return true;
      }
      
      // 去除行首空格
      size_t start = line.find_first_not_of(" \t");
      if (start == std::string::npos) {
          return true; // 全是空格
      }
      
      // 检查是否为注释行
      return line[start] == '#';
  }

  bool getQRCodePose(int tag_id, QRCodeMapPose& result) const {
      auto it = qrcode_map_poses_.find(tag_id);
      if (it != qrcode_map_poses_.end()) {
          result = it->second;
          return true;
      }
      return false;
  }

  void publishQRCodeMarkers() {
    if (qrcode_map_poses_.empty()) {
        std::cout << "No QR code poses to publish." << std::endl;
        return;
    }
    
    visualization_msgs::MarkerArray marker_array;
    
    int marker_id = 0;
    for (const auto& pair : qrcode_map_poses_) {
        const QRCodeMapPose& qr_pose = pair.second;
        int tag_id = qr_pose.tag_id;
        
        // 1. 创建文字标签（显示二维码ID）
        visualization_msgs::Marker text_marker;
        text_marker.header.frame_id = "world";
        text_marker.header.stamp = ros::Time::now();
        text_marker.ns = "qrcode_text";
        text_marker.id = marker_id++;
        text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::Marker::ADD;
        
        // 设置文字位置（在二维码位置上方0.3米处）
        text_marker.pose.position.x = qr_pose.position.x();
        text_marker.pose.position.y = qr_pose.position.y();
        text_marker.pose.position.z = qr_pose.position.z() + 0.3;
        text_marker.pose.orientation.x = 0.0;
        text_marker.pose.orientation.y = 0.0;
        text_marker.pose.orientation.z = 0.0;
        text_marker.pose.orientation.w = 1.0;
        
        // 设置文字内容和样式
        text_marker.text = "QR-" + std::to_string(tag_id);
        text_marker.scale.z = 0.2;  // 文字高度
        text_marker.color.r = 1.0;  // 红色文字
        text_marker.color.g = 1.0;
        text_marker.color.b = 0.0;
        text_marker.color.a = 1.0;  // 不透明度
        
        text_marker.lifetime = ros::Duration(0);  // 永久显示
        marker_array.markers.push_back(text_marker);
        
        // 2. 创建箭头（显示二维码朝向）
        visualization_msgs::Marker arrow_marker;
        arrow_marker.header.frame_id = "world";
        arrow_marker.header.stamp = ros::Time::now();
        arrow_marker.ns = "qrcode_arrow";
        arrow_marker.id = marker_id++;
        arrow_marker.type = visualization_msgs::Marker::ARROW;
        arrow_marker.action = visualization_msgs::Marker::ADD;
        
        // 设置箭头位置和方向
        arrow_marker.pose.position.x = qr_pose.position.x();
        arrow_marker.pose.position.y = qr_pose.position.y();
        arrow_marker.pose.position.z = qr_pose.position.z();
        
        // 使用二维码的四元数方向
        arrow_marker.pose.orientation.x = qr_pose.orientation.x();
        arrow_marker.pose.orientation.y = qr_pose.orientation.y();
        arrow_marker.pose.orientation.z = qr_pose.orientation.z();
        arrow_marker.pose.orientation.w = qr_pose.orientation.w();
        
        // 设置箭头大小和颜色
        arrow_marker.scale.x = 0.4;  // 箭头长度
        arrow_marker.scale.y = 0.05; // 箭头宽度
        arrow_marker.scale.z = 0.05; // 箭头高度
        arrow_marker.color.r = 0.0;  // 绿色箭头
        arrow_marker.color.g = 1.0;
        arrow_marker.color.b = 0.0;
        arrow_marker.color.a = 0.8;  // 半透明
        
        arrow_marker.lifetime = ros::Duration(0);  // 永久显示
        marker_array.markers.push_back(arrow_marker);
        
        // 3. 创建立方体（表示二维码位置）
        visualization_msgs::Marker cube_marker;
        cube_marker.header.frame_id = "world";
        cube_marker.header.stamp = ros::Time::now();
        cube_marker.ns = "qrcode_cube";
        cube_marker.id = marker_id++;
        cube_marker.type = visualization_msgs::Marker::CUBE;
        cube_marker.action = visualization_msgs::Marker::ADD;
        
        // 设置立方体位置（与二维码相同位置，但在地面上方一点点）
        cube_marker.pose.position.x = qr_pose.position.x();
        cube_marker.pose.position.y = qr_pose.position.y();
        cube_marker.pose.position.z = qr_pose.position.z() + 0.025;  // 略微抬高
        cube_marker.pose.orientation.x = qr_pose.orientation.x();
        cube_marker.pose.orientation.y = qr_pose.orientation.y();
        cube_marker.pose.orientation.z = qr_pose.orientation.z();
        cube_marker.pose.orientation.w = qr_pose.orientation.w();
        
        // 设置立方体大小和颜色
        cube_marker.scale.x = 0.5;  // 二维码尺寸
        cube_marker.scale.y = 0.5;
        cube_marker.scale.z = 0.05; // 很薄的立方体
        cube_marker.color.r = 0.0;  // 蓝色立方体
        cube_marker.color.g = 0.0;
        cube_marker.color.b = 1.0;
        cube_marker.color.a = 0.5;  // 半透明
        
        cube_marker.lifetime = ros::Duration(0);  // 永久显示
        marker_array.markers.push_back(cube_marker);

    }
    
    // 发布MarkerArray
    pub_qrcode_markers_.publish(marker_array);
  }

  void publishQRCodeMarkersTimer(const ros::TimerEvent& event) {
      // 只有在有二维码数据时才发布
      if (!qrcode_map_poses_.empty()) {
          publishQRCodeMarkers();
      }
  }



};

int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);
  ros::init(argc, argv, "LOC");
  ROS_INFO("\033[1;32m----> LOC Started.\033[0m");

  std::cout << "ROOT_DIR: " << root_dir << std::endl;

  map_location *lol = new map_location();
  std::thread opt_thread(&map_location::run, lol);

  ros::spin();

  return 0;
}
