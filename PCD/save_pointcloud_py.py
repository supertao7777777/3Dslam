#!/usr/bin/env python3
import rospy
import os
import threading
from livox_ros_driver2.msg import CustomMsg
from datetime import datetime

# ==================== 配置参数 ====================
# PCD文件保存路径（请根据实际需求修改）
PCD_SAVE_PATH = "/home/li/ws_3D/PCD/source/"  # 请替换为您的实际路径

# 保存触发条件（可选）
SAVE_INTERVAL = 1.0  # 保存间隔（秒），设置为0表示每帧都保存
MAX_POINTS_THRESHOLD = 10000  # 当点云数量超过此阈值时才保存

# ==================== 全局变量 ====================
point_cloud_data = []
last_save_time = rospy.Time(0)
lock = threading.Lock()
should_exit = False  # 新增：退出标志
subscriber = None    # 新增：订阅对象引用
save_thread = None   # 新增：保存线程引用

# ==================== 点云回调函数 ====================
def point_cloud_callback(msg):
    global point_cloud_data, last_save_time, should_exit, subscriber
    
    # 如果已触发退出，直接返回
    if should_exit:
        return
    
    current_time = rospy.Time.now()
    points = []
    
    # 提取点云数据
    for point in msg.points:
        # 添加x, y, z坐标和反射强度
        points.append([point.x, point.y, point.z, point.reflectivity])
    
    # 如果点云数量太少，跳过保存
    if len(points) < 100:
        rospy.logdebug("点云数据量不足，跳过保存")
        return
    
    # 检查保存条件
    if (current_time - last_save_time).to_sec() < SAVE_INTERVAL and SAVE_INTERVAL > 0:
        return
    
    with lock:
        # 如果已触发退出，直接返回
        if should_exit:
            return
            
        point_cloud_data = points
        last_save_time = current_time
        
        # 在新线程中保存数据，避免阻塞回调函数
        global save_thread
        save_thread = threading.Thread(target=save_pcd_data)
        save_thread.start()
        
        # 取消订阅，避免重复触发
        if subscriber is not None:
            subscriber.unregister()
            subscriber = None
            rospy.loginfo("已取消订阅，避免重复保存")

# ==================== PCD文件保存函数 ====================
def save_pcd_data():
    global point_cloud_data, should_exit
    
    if not point_cloud_data:
        return
    
    # 确保保存目录存在
    if not os.path.exists(PCD_SAVE_PATH):
        raise FileNotFoundError(f"目录不存在: {PCD_SAVE_PATH}")
    
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(PCD_SAVE_PATH, f"point_cloud_relocazation.pcd")
    
    try:
        with open(filename, 'w') as pcd_file:
            # 写入PCD文件头
            pcd_file.write("# .PCD v.7 - Point Cloud Data file format\n")
            pcd_file.write("VERSION .7\n")
            pcd_file.write("FIELDS x y z intensity\n")
            pcd_file.write("SIZE 4 4 4 4\n")
            pcd_file.write("TYPE F F F F\n")
            pcd_file.write("COUNT 1 1 1 1\n")
            pcd_file.write(f"WIDTH {len(point_cloud_data)}\n")
            pcd_file.write("HEIGHT 1\n")
            pcd_file.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            pcd_file.write(f"POINTS {len(point_cloud_data)}\n")
            pcd_file.write("DATA ascii\n")
            
            # 写入点云数据
            for point in point_cloud_data:
                pcd_file.write(f"{point[0]} {point[1]} {point[2]} {point[3]}\n")
        
        rospy.loginfo(f"PCD文件保存成功: {filename}")
        
        # 设置退出标志
        global should_exit
        should_exit = True
        rospy.loginfo("保存完成，准备退出节点...")
        
    except IOError as e:
        rospy.logerr(f"保存PCD文件失败: {str(e)}")

# ==================== 主函数 ====================
def main():
    global subscriber, should_exit
    
    rospy.init_node('livox_pcd_saver', anonymous=True)
    
    # 显示保存路径信息
    rospy.loginfo(f"PCD文件将保存到: {PCD_SAVE_PATH}")
    
    if not os.path.exists(PCD_SAVE_PATH):
        raise FileNotFoundError(f"目录不存在: {PCD_SAVE_PATH}")
    
    # 订阅Livox点云话题
    subscriber = rospy.Subscriber("/livox/lidar", CustomMsg, point_cloud_callback, queue_size=10)
    
    rospy.loginfo("开始监听/livox/lidar话题，准备保存点云数据...")
    rospy.loginfo("按Ctrl+C可手动退出程序")
    
    # 使用循环检查退出条件
    rate = rospy.Rate(10)  # 10Hz检查频率
    while not rospy.is_shutdown() and not should_exit:
        rate.sleep()
    
    # 等待保存线程完成
    if save_thread is not None:
        save_thread.join(timeout=1.0)
        if save_thread.is_alive():
            rospy.logwarn("保存线程未正常结束，强制退出")
        else:
            rospy.loginfo("保存线程已完成")
    
    rospy.loginfo("节点安全退出")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
