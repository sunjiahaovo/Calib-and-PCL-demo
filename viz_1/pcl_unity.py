from ast import For
import open3d as o3d
import numpy as np
import os
import imageio
import copy

base_dir = "data/cabin-0.5/"
n_imgs = 1


pcd_combined = o3d.geometry.PointCloud()

FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0,0, 0])
# FOR1.transform([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
# FOR1.transform([[0, 0, -1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])

T1 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
T2 = np.array([[0, 0, -1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
T = T2@ T1
print(T)
FOR1.transform(T.tolist())
FOR1.transform(np.linalg.inv(T).tolist())

for i in np.arange(0,40,1):
    #加载图片
    print(i,os.path.join(base_dir,f'{i}main.png'))
    # rgb = imageio.imread(os.path.join(base_dir,f'{i}main.png'))[:,:,:3]
    # imageio.imwrite(os.path.join(base_dir,f'{i}main_v.png'),rgb)
    color_raw  = o3d.io.read_image(os.path.join(base_dir,f'{i}main_v.png'))
    # depth = imageio.imread(os.path.join(base_dir,  f'{i}depth.png'))[:,:,0]/255.0
    # depth[depth==1] = 0
    # depth = (depth*6*1000).astype(np.uint16)
    # imageio.imwrite(os.path.join(base_dir,  f'{i}depth_v.png'),depth)
    depth_raw  = o3d.io.read_image(os.path.join(base_dir,  f'{i}depth_v.png'))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)

    
    #内参赋值
    fx, fy, cx, cy = 600, 600, 400, 400
    width = 800
    height = 800

    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image( rgbd_image,intrinsic)
    # pcd.transform([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
    # transform
    cam_pose = np.loadtxt(os.path.join(base_dir,f'{i}.txt'), delimiter = "\t")
    print(cam_pose)

    
    print("before ",cam_pose)
    T_b2u = np.array([[-1,0,0,0],
                      [0,1,0,0],
                      [0,0,1,0], 
                      [0,0,0,1]])
    cam_pose =  T_b2u @ cam_pose
    print("after ",cam_pose)
    # pcd.transform([[1, 0, 0, -t[1]], [0, 1, 0, t[2]], [0, 0, 1, -t[0]], [0, 0, 0, 1]])
    pcd.transform(cam_pose)

    ## 拼接
    pcd_combined += pcd
# o3d.io.write_point_cloud("pcd_22.pcd",pcd_combined)

o3d.visualization.draw_geometries([pcd_combined,FOR1])