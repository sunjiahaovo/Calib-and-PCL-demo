import open3d as o3d
import numpy as np
import os
import imageio
import imageio
import cv2

base_dir = "data2/monosdf/room"

# mask = np.load(os.path.join(base_dir,  '000037_mask.npy'))
# cv2.imshow("mask", mask) 
# cv2.waitKey(0)

depth = imageio.imread(os.path.join(base_dir,  '188depth.png'))[:,:]/255
# depth = np.load(os.path.join(base_dir,  '188main_depth.npy'))
depth = cv2.resize(depth,(800,800))
# depth = depth[50:750,50:750]
# depth = np.load(os.path.join(base_dir,  '000037_depth.npy'))
print(depth)
print(depth.shape)

def depth2xyz(depth_map,depth_cam_matrix,flatten=False,depth_scale=1):
    fx,fy = depth_cam_matrix[0,0],depth_cam_matrix[1,1]
    cx,cy = depth_cam_matrix[0,2],depth_cam_matrix[1,2]
    H, W = depth_map.shape[0],depth_map.shape[1]
    h,w=np.mgrid[0:H,0:W]
    z=depth_map/depth_scale
    x=(w-cx)*z/fx
    y=(h-cy)*z/fy
    xyz=np.dstack((x,y,z)) if flatten==False else np.dstack((x,y,z)).reshape(-1,3)
    print(xyz.shape)
    xyz = xyz.reshape(H*W,3)
    # xyz = xyz[~(xyz==0).all(1)]
    
    return xyz
    

pcd_combined = o3d.geometry.PointCloud()
depth_cam_matrix = np.array([[600, 0,  400],
                                [0,   600, 400],
                                [0,   0,    1]])

# depth_cam_matrix = np.array([[288, 0,  192],
#                                 [0,   288, 192],
#                                 [0,   0,    1]])


xyz = depth2xyz(depth, depth_cam_matrix)
print(xyz,xyz.shape)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz[::1])
# pcd_new = o3d.geometry.PointCloud.remove_radius_outlier(pcd, 10, 0.01)
o3d.visualization.draw_geometries([pcd])

