import open3d as o3d
import numpy as np
import os
import cv2

base_dir = "data/box_24/"
n_imgs = 3


pcd_combined = o3d.geometry.PointCloud()

offset_T = np.loadtxt("offset_1.txt")


index = np.array([0,1,2,3,4,5,6,7,9,10,14,15,16,17])

for i in range(23):
    #加载图片
    color_raw  = o3d.io.read_image(os.path.join(base_dir,"%d_main.png"%(i)))
    depth_raw  = o3d.io.read_image(os.path.join(base_dir, "%d_depth.png"%(i)))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
    depth_im  = cv2.imread(os.path.join(base_dir, "%d_depth.png"%(i)))

    
    #内参赋值
    fx, fy, cx, cy = 899.08, 898.04, 373.99, 651.06
    width = 720
    height = 1280

    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    # intrinsic = o3d.camera.PinholeCameraIntrinsic(height, width, fy, fx, cy, cx)


    pcd = o3d.geometry.PointCloud.create_from_rgbd_image( rgbd_image,intrinsic)
    pcd.transform([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

    # transform
    uav_pose = np.loadtxt(os.path.join(base_dir,"%d_pose.txt"%(i)), delimiter = ",")
    trans = np.linalg.multi_dot([np.linalg.inv(offset_T), uav_pose, offset_T])
    pcd.transform(trans)

    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pcd_combined += pcd
o3d.io.write_point_cloud("pcd_22.ply",pcd_combined)
o3d.io.write_point_cloud("pcd_22.pcd",pcd_combined)

o3d.visualization.draw_geometries([pcd_combined])