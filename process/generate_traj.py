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

views = []
object_center = np.array([0,0,0.7])
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Open3D', width=800, height=600, left=50, top=50, visible=True)
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=np.array([0.0, 0.0, 0.0])) # 直接就是坐标轴，没有点的

indxs = [i for i in range(32)]
indxs = np.delete(np.array(indxs),13)
for i in indxs:
    if i == 0:
        vis.reset_view_point(True)
    pose = np.loadtxt(os.path.join("data1","obj_v16",str(i)+'_pose.txt'),delimiter=",")
    x,y,z = pose[0,3],pose[1,3],pose[2,3]
    z = z + 0.1 + np.random.randn()*0.2
    if y > 1.2:
        y = 1.2-np.random.uniform(0.,0.2)
    if y < -1.2:
       y = -1.2+np.random.uniform(0.,0.2)
       
    if z < 0.5:
        z =0.5+np.random.uniform(0.,0.1)
    if z > 1.3:
        z =1.3-np.random.uniform(0.,0.1)
        
        
    dx,dy,dz =  object_center[0]-x,object_center[1]-y,object_center[2]-z
    v= np.arctan2(dy,dx) + np.random.randn()*10/180*np.pi
    print(x,y,z,v/np.pi*180)
    views.append([x,y,z,v])
    pose = get_pose([x,y,z],0,v)
   
    cam_mesh = copy.deepcopy(mesh).transform(pose)
    vis.add_geometry(cam_mesh)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.02)
np.savetxt("views.txt",np.array(views))
vis.run()