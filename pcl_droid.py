import open3d as o3d
import numpy as np
import os
from math import degrees
from scipy.spatial.transform import Rotation as R

base_dir = "data/transfer_data_realsense_depth"
n_imgs = 1

pcd_combined = o3d.geometry.PointCloud()

FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

# T1 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
# T2 = np.array([[0, 0, -1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
# T = T2@ T1
# print(T)
# FOR1.transform(T.tolist())
# FOR1.transform(np.linalg.inv(T).tolist())



for i in np.arange(0,57,1):
    #加载图片
    color_raw  = o3d.io.read_image(os.path.join(base_dir, 'frame{:05d}.png'.format(i+1)))
    depth_raw  = o3d.io.read_image(os.path.join(base_dir, 'depth{:05d}.png'.format(i+1)))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)


    #内参赋值
    fx, fy, cx, cy = 607.89, 607.93, 320.26, 238.11
    width = 640
    height = 480


    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    # pcd.transform([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
    # transform
    uav_pose = np.loadtxt(os.path.join(base_dir,f'{i}_pose.txt'), delimiter = " ")
    trans = np.concatenate((uav_pose, np.array([[0,0,0,1]])), axis=0)
    
    print("before ",trans)
    # T_b2o = np.array([[0,0,-1,0],
    #                   [-1,0,0,0],
    #                   [0,1,0,0],
    #                   [0,0,0,1]])
    # trans = np.linalg.inv(T_b2o)@trans@ T_b2o
    
    
    
    print("after ",trans)
    # pcd.transform([[1, 0, 0, -t[1]], [0, 1, 0, t[2]], [0, 0, 1, -t[0]], [0, 0, 0, 1]])
    pcd.transform(trans)
    

    ## 拼接
    pcd_combined += pcd
# o3d.io.write_point_cloud("obj_pnp_v5_t265_coarse_offset.ply",pcd_combined)
# o3d.io.write_point_cloud("pcd_22.pcd",pcd_combined)

o3d.visualization.draw_geometries([pcd_combined,FOR1])