#!/usr/bin/env python3
import os
import shutil
from datetime import datetime

def replace_pcd_file(source_dir, target_dir, pcd_filename):
    """
    用源目录中的指定PCD文件替换目标目录中的同名文件
    
    参数:
        source_dir: 源目录路径
        target_dir: 目标目录路径  
        pcd_filename: 要替换的PCD文件名
    """
    # 构建完整路径
    source_path = os.path.join(source_dir, pcd_filename)
    target_path = os.path.join(target_dir, pcd_filename)
    
    # 检查源文件是否存在
    if not os.path.exists(source_path):
        log_message(f"错误: 源文件不存在 - {source_path}")
        return False
    
    # 检查目标目录是否存在
    if not os.path.exists(target_dir):
        log_message(f"错误: 目标目录不存在 - {target_dir}")
        return False
    
    # 检查目标文件是否存在
    target_exists = os.path.exists(target_path)
    
    try:
        # 获取源文件的创建时间
        src_stat = os.stat(source_path)
        src_ctime = datetime.fromtimestamp(src_stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
        
        log_message(f"源文件创建时间: {src_ctime}")
        
        # 如果目标文件存在，获取其创建时间
        if target_exists:
            target_stat = os.stat(target_path)
            target_ctime = datetime.fromtimestamp(target_stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
            log_message(f"原目标文件创建时间: {target_ctime}")
        
        # 复制/替换文件
        shutil.copy2(source_path, target_path)
        
        # 获取新目标文件的创建时间
        new_target_stat = os.stat(target_path)
        new_target_ctime = datetime.fromtimestamp(new_target_stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
        log_message(f"新目标文件创建时间: {new_target_ctime}")
        
        if target_exists:
            log_message(f"成功替换文件: {pcd_filename}")
        else:
            log_message(f"成功创建文件: {pcd_filename}")
            
        return True
        
    except Exception as e:
        log_message(f"替换过程中发生错误: {str(e)}")
        return False

def log_message(message):
    """统一的日志输出函数"""
    try:
        import rospy
        rospy.loginfo(message)
    except ImportError:
        print(f"[INFO] {message}")

if __name__ == '__main__':
    try:
        import rospy
        rospy.init_node('pcd_file_replacer', anonymous=True)
    except ImportError:
        print("ROS未安装，以标准Python模式运行")
    
    # 设置路径和文件名 - 修改为你的实际值
    source_directory = '/home/li/ws_3D/src/FAST_LIO_SAM/result'  # 替换为你的源目录
    target_directory = '/home/li/ws_3D/PCD/target'  # 替换为你的目标目录
    pcd_file_name = 'GlobalMap.pcd'  # 替换为你要替换的PCD文件名
    
    log_message("开始PCD文件替换操作...")
    
    # 执行替换操作
    success = replace_pcd_file(source_directory, target_directory, pcd_file_name)
    
    if success:
        log_message("文件替换操作完成")
    else:
        log_message("文件替换操作失败")