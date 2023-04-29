import open3d as o3d
import numpy as np
import os
import imageio
import copy
from math import degrees
from scipy.spatial.transform import Rotation as R

base_dir = "data_for_pcl/jiahao_flight1/"
n_imgs = 1

# for i in np.arange(0,n_imgs,1):
#     depth_img =imageio.imread(os.path.join(base_dir,  f'{i}_depth.png'))
#     print("depth_img ",depth_img.shape)
#     # depth_img = (depth_img*2.3942470545582677).astype(np.uint16)
#     # imageio.imwrite(os.path.join(base_dir,  f'{i}_depth_scale.png'),depth_img)
#     # print(depth_img.shape)

pcd_combined = o3d.geometry.PointCloud()
offset_T = np.loadtxt("data/offset_pnp_v2.txt")

FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0,0, 0])

# T1 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
# T2 = np.array([[0, 0, -1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
# T = T2@ T1
# print(T)
# FOR1.transform(T.tolist())
# FOR1.transform(np.linalg.inv(T).tolist())

# load xyzw
pose_vec = np.loadtxt("data_for_pcl/jiahao_flight1.txt")[:]
t265_time = pose_vec[:,0]
t265_t = pose_vec[:,1:4]
t265_q = pose_vec[:,4:8]

init_time = t265_time[0]
for i in range(t265_time.shape[0]):
    t265_time[i] = t265_time[i] - init_time

# set up rotation matrix
t265_R = []

# set up trans matrix
t265_T = np.zeros((4, 4))

# q2R2T
for i in range(t265_time.shape[0]):
    # 创建旋转矩阵
    t265_R = R.from_quat([t265_q[i,0],t265_q[i,1],t265_q[i,2],t265_q[i,3]])
    t265_T[:3, :3] = t265_R.as_matrix()
    t265_T[:3, 3] = t265_t[i, :3]
    t265_T[3, 3] = 1
    np.savetxt(os.path.join(base_dir, f'{i}_t265_pose.txt'), t265_T)



for i in np.arange(0,2390,10):
    #加载图片
    print(os.path.join(base_dir,f'{i}_main.png'))
    color_raw  = o3d.io.read_image(os.path.join(base_dir,f'{i}_main.png'))
    depth_raw  = o3d.io.read_image(os.path.join(base_dir,  f'{i}_depth.png'))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)


    
    #内参赋值
    # fx, fy, cx, cy = 911.87, 911.87, 640, 360
    # fx, fy, cx, cy = 385.7544860839844, 385.7544860839844, 323.1204833984375, 236.7432098388672
    
    #d435i内参
    fx, fy, cx, cy = 607.89, 607.93, 320.26, 238.11
    width = 640
    height = 480

    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


    pcd = o3d.geometry.PointCloud.create_from_rgbd_image( rgbd_image,intrinsic)
    # pcd.transform([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
    # transform
    uav_pose = np.loadtxt(os.path.join(base_dir,f'{i}_t265_pose.txt'), delimiter = " ")
    # print(uav_pose)
    # trans = uav_pose @ offset_T
    trans = np.linalg.multi_dot([np.linalg.inv(offset_T), uav_pose, offset_T])
    # trans = uav_pose
    
    print("before ",trans)
    T_b2o = np.array([[0,0,-1,0],
                      [-1,0,0,0],
                      [0,1,0,0],
                      [0,0,0,1]])
    trans = np.linalg.inv(T_b2o)@trans@ T_b2o
    
    
    
    print("after ",trans)
    # pcd.transform([[1, 0, 0, -t[1]], [0, 1, 0, t[2]], [0, 0, 1, -t[0]], [0, 0, 0, 1]])
    pcd.transform(trans)
    

    ## 拼接
    pcd_combined += pcd
# o3d.io.write_point_cloud("obj_pnp_v5_t265_coarse_offset.ply",pcd_combined)
# o3d.io.write_point_cloud("pcd_22.pcd",pcd_combined)

o3d.visualization.draw_geometries([pcd_combined,FOR1])