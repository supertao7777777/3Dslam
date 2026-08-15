#!/usr/bin/env python3
from __future__ import annotations

import os
import threading
from datetime import datetime

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

# ==================== 配置参数（可通过ROS参数覆盖） ====================
# ~topic:           订阅的话题名（默认: /scan）
# ~save_dir:        PCD保存目录（默认: 当前脚本目录下 pcd_out）
# ~filename:        保存文件名（默认: scan_frame.pcd）
# ~use_timestamp:   文件名是否追加时间戳（默认: true）
# ~min_points:      点数少于该值则不保存（默认: 50）

# ==================== 全局变量 ====================
_lock = threading.Lock()
_should_exit = False
_subscriber = None
_save_thread = None

_points_cache = None          # list[tuple]
_pcd_fields = None            # list[str]
_save_path = None             # str
_min_points = 0               # int


def _choose_fields(msg: PointCloud2):
    names = {f.name for f in msg.fields}

    fields = ["x", "y", "z"]
    if not all(f in names for f in fields):
        missing = [f for f in fields if f not in names]
        raise ValueError(f"PointCloud2缺少必要字段: {missing}")

    if "intensity" in names:
        fields.append("intensity")
    elif "reflectivity" in names:
        # 有些点云会用reflectivity表达强度，这里统一写成intensity列
        fields.append("reflectivity")

    return fields


def _write_pcd_ascii(path: str, fields: list[str], points: list[tuple]):
    # 将reflectivity列写成PCD里的intensity列名，兼容常见PCL读取习惯
    pcd_fields = ["x", "y", "z"]
    if len(fields) == 4:
        pcd_fields.append("intensity")

    with open(path, "w") as f:
        f.write("# .PCD v.7 - Point Cloud Data file format\n")
        f.write("VERSION .7\n")
        f.write(f"FIELDS {' '.join(pcd_fields)}\n")
        f.write(f"SIZE {' '.join(['4'] * len(pcd_fields))}\n")
        f.write(f"TYPE {' '.join(['F'] * len(pcd_fields))}\n")
        f.write(f"COUNT {' '.join(['1'] * len(pcd_fields))}\n")
        f.write(f"WIDTH {len(points)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\n")
        f.write("DATA ascii\n")

        if len(fields) == 3:
            for x, y, z in points:
                f.write(f"{float(x)} {float(y)} {float(z)}\n")
        else:
            for x, y, z, intensity in points:
                f.write(f"{float(x)} {float(y)} {float(z)} {float(intensity)}\n")


def _save_once():
    global _points_cache, _pcd_fields, _should_exit

    with _lock:
        points = _points_cache
        fields = _pcd_fields

    if not points:
        return

    try:
        _write_pcd_ascii(_save_path, fields, points)
        rospy.loginfo("PCD文件保存成功: %s (points=%d, fields=%s)", _save_path, len(points), fields)
        _should_exit = True
        rospy.loginfo("保存完成，准备退出节点...")
    except Exception as e:
        rospy.logerr("保存PCD文件失败: %s", str(e))


def _callback(msg: PointCloud2):
    global _points_cache, _pcd_fields, _subscriber, _save_thread

    if _should_exit:
        return

    try:
        fields = _choose_fields(msg)
        pts = list(pc2.read_points(msg, field_names=fields, skip_nans=True))
    except Exception as e:
        rospy.logerr("解析PointCloud2失败: %s", str(e))
        return

    if len(pts) < _min_points:
        rospy.logwarn("点数(%d) < min_points(%d)，跳过保存", len(pts), _min_points)
        return

    with _lock:
        if _should_exit:
            return
        _points_cache = pts
        _pcd_fields = fields

        _save_thread = threading.Thread(target=_save_once)
        _save_thread.daemon = True
        _save_thread.start()

        if _subscriber is not None:
            _subscriber.unregister()
            _subscriber = None
            rospy.loginfo("已取消订阅，避免重复保存")


def main():
    global _subscriber, _save_path, _min_points

    rospy.init_node("scan_pointcloud2_pcd_saver", anonymous=True)

    topic = rospy.get_param("~topic", "/scan")
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pcd_out")
    save_dir = rospy.get_param("~save_dir", default_dir)
    filename = rospy.get_param("~filename", "scan_frame.pcd")
    use_timestamp = bool(rospy.get_param("~use_timestamp", True))
    _min_points = int(rospy.get_param("~min_points", 50))

    os.makedirs(save_dir, exist_ok=True)

    if use_timestamp:
        stem, ext = os.path.splitext(filename)
        if not ext:
            ext = ".pcd"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{stem}_{ts}{ext}"

    _save_path = os.path.join(save_dir, filename)

    rospy.loginfo("订阅PointCloud2话题: %s", topic)
    rospy.loginfo("PCD将保存到: %s", _save_path)
    rospy.loginfo("min_points=%d, use_timestamp=%s", _min_points, str(use_timestamp))

    _subscriber = rospy.Subscriber(topic, PointCloud2, _callback, queue_size=1)

    rate = rospy.Rate(20)
    while not rospy.is_shutdown() and not _should_exit:
        rate.sleep()

    if _save_thread is not None:
        _save_thread.join(timeout=2.0)

    rospy.loginfo("节点退出")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
