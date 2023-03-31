import open3d as o3d
import cv2
import numpy as np
import matplotlib
import copy
import time
import os
from scipy.spatial.transform import Rotation as R


def polygon():
    #绘制顶点
    polygon_points = []
    for j in range(36):
        theta = j*10/180*np.pi
        x = -1.7*np.cos(theta)
        y = 1.3*np.sin(theta)
        z = 0.6
        polygon_points.append([x,y,z])
    polygon_points = np.array(polygon_points)
    lines = [[i,i+1] for i in range(len(polygon_points)-1)] #连接的顺序，封闭链接
    color = [[1, 0, 0] for i in range(len(polygon_points))] 
    #添加顶点，点云
    points_pcd = o3d.geometry.PointCloud()
    points_pcd.points = o3d.utility.Vector3dVector(polygon_points)
    points_pcd.paint_uniform_color([0, 0.3, 0]) #点云颜色
 
    #绘制线条
    lines_pcd = o3d.geometry.LineSet()
    lines_pcd.lines = o3d.utility.Vector2iVector(lines)
    lines_pcd.colors = o3d.utility.Vector3dVector(color) #线条颜色
    lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)
 
    return lines_pcd, points_pcd


vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Open3D', width=800, height=600, left=50, top=50, visible=True)
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=np.array([0.0, 0.0, 0.0])) # 直接就是坐标轴，没有点的

# lines, points = polygon()
# vis.add_geometry(lines)
# vis.add_geometry(points)


for i in range(0,200,1):
    if i == 0:
        vis.reset_view_point(True)
    pose = np.loadtxt(os.path.join("data1","room_coverall",str(i)+'.txt'),delimiter="\t")
    cam_mesh = copy.deepcopy(mesh).transform(pose)
    vis.add_geometry(cam_mesh)
    
    dir_noise = np.random.randn(2)*1
    t_noise = np.random.randn(3)*0.0
    # print(dir_noise,t_noise)
    r4 = R.from_euler('ZYX', [0,  dir_noise[0],  dir_noise[1]], degrees=True)
    rm = r4.as_matrix()
    # print("rm = ", rm)
    pose[:3,:3] = rm @ pose[:3,:3]
    pose[:3,3] = pose[:3,3] + t_noise
    print("pose = ", pose)

    cam_mesh = copy.deepcopy(mesh).transform(pose)
    vis.add_geometry(cam_mesh)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.02)
vis.run()

