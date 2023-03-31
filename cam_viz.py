import open3d as o3d
import cv2
import numpy as np
import matplotlib
import copy
import time
import os
from scipy.spatial.transform import Rotation as R

# Real Scene
def get_pose(location,u,v):
    v = v/np.pi*180
    u = -u/np.pi*180 
    r = R.from_euler('ZYX', [v, u, 0], degrees=True)
    pose = np.zeros((4,4))
    pose[0:3,0:3] = r.as_matrix()
    pose[0:3,3] = np.array(location)
    pose[3,3]=1
    # print("pose = ", pose)
    return pose


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

lines, points = polygon()
vis.add_geometry(lines)
vis.add_geometry(points)


offset_T = np.loadtxt("offsets/offset_optitrack_v20.txt")
for i in range(0,31,1):
    if i == 0:
        vis.reset_view_point(True)
    pose = np.loadtxt(os.path.join("data1","obj_v27",str(i)+'_pose.txt'),delimiter=",")
    pose = np.linalg.multi_dot([np.linalg.inv(offset_T), pose, offset_T])
    

    cam_mesh = copy.deepcopy(mesh).transform(pose)
    vis.add_geometry(cam_mesh)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.02)
vis.run()

# aabb_bnds = np.array([[-1.5,1.5],[-0.9,0.9],[0.1,1.2]])    ## tsdf uncertainty包围盒
# aoi_bnds = np.array([[-2.2,2.2],[-1.2,1.3],[0.1,3]])    ## tsdf uncertainty包围盒
# object_center = np.array([0,0,0.7])
# k = 0
# for i in range(0,15,1):
#     if i == 0:
#         vis.reset_view_point(True)
#     n = 0
#     while n<2:  
#         is_valid = True 
#         theta = i*25/180*np.pi
#         theta = theta + np.random.randn()*10/180*np.pi
#         x = -1.8*np.cos(theta) + np.random.randn()*0.5
#         y = 1.05*np.sin(theta) + np.random.randn()*0.05
#         z = np.random.uniform(0.4,1.0)+ np.random.randn()*0.3
        
#         ## 如果在uncertainty包围盒里面
#         if aabb_bnds[0,0]<x<aabb_bnds[0,1] and \
#             aabb_bnds[1,0]<y<aabb_bnds[1,1]:
#             is_valid = False
        
#         ## 如果在aoi包围盒外面
#         if not (aoi_bnds[0,0]<x<aoi_bnds[0,1] and \
#             aoi_bnds[1,0]<y<aoi_bnds[1,1]) :
#             is_valid = False
            
#         if z< 0.4 or z > 1.0:
#             is_valid = False
#         if not is_valid:
#             continue
#         n = n + 1
#         print(x,y,z)        

#         dx,dy,dz =  object_center[0]-x,object_center[1]-y,object_center[2]-z
#         v = np.arctan2(dy,dx) + np.random.randn()*10/180*np.pi
#         pose = get_pose([x,y,z],0,v)
#         # print(pose)
        
#         np.savetxt(os.path.join("data1/obj_v10_traj",f'{k}_pose.txt'),pose)
#         k = k + 1
        

#         cam_mesh = copy.deepcopy(mesh).transform(pose)
#         vis.add_geometry(cam_mesh)
#         vis.poll_events()
#         vis.update_renderer()
#         time.sleep(0.02)
# vis.run()

    


