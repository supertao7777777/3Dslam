------建图系统--------
------FAST-LIO-SAM-----
数据集：mid-360
运行步骤：
roslaunch fast_lio_sam mapping_mid360.launch 
rosbag play
（建图完成，保存地图，关键帧，二维码）
另起一个终端,source
rosservice call /save_pose "resolution: 0.0
destination: ''" 

rosservice call /save_map "resolution: 0.0
destination: ''" 

rosservice call /fast_lio_sam/save_qrcode "resolution: 0.0
destination: ''" 

-------定位系统---------
-------Lidar_IMU_Localization--------
1.roslaunch lio_localization run_loc.launch 

-------todo------
1.外推器
2.地图扩建

