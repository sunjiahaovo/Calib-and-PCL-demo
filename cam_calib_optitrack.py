import cv2
import numpy as np
import os
import time
from numpy.linalg.linalg import _multidot_dispatcher
import icp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图

base_dir = "data/cam_v6" # 测试数据所在文件夹
base_dir = "data1/calib_v2" # 测试数据所在文件夹
base_dir = "data1/calib_v10" # 测试数据所在文件夹
base_dir = "data2/calib_v1" # 测试数据所在文件夹

n = 0
N = 15# 测试点的数量

K = np.array([[911.87, 0, 640],
            [0, 911.87, 360],
            [0, 0, 1]])

# K = np.array([[911.85, 0, 640],
#             [0, 911.89, 357.16],
#             [0, 0, 1]])


# K = np.array([[898.04, 0, 651.06],
#             [0, 899.08, 373.99],
#             [0, 0, 1]])



## 读取数据
color_image = cv2.cvtColor(cv2.imread(os.path.join(base_dir,"%d_main.png"%(n))), cv2.COLOR_BGR2RGB)
print(color_image.shape)
depth_im = cv2.imread(os.path.join(base_dir,"%d_depth.png"%(n)),-1).astype(float)/1000.0
marker_pixels = np.loadtxt(os.path.join(base_dir,"%d_pixels.txt"%(n)),delimiter=" ").astype(int)
uav_pose = np.loadtxt(os.path.join(base_dir,"%d_pose.txt"%(n)),delimiter=",")
marker_cords = np.loadtxt(os.path.join(base_dir,"%d_markers.txt"%(n)),delimiter=" ")
marker_cords = np.concatenate((marker_cords[:,0][:,np.newaxis],-marker_cords[:,2][:,np.newaxis],marker_cords[:,1][:,np.newaxis]),axis = 1)
marker_pixels_cal=marker_pixels[:,[1,0]]
marker_cords = marker_cords[0:N,:]

## 加depth pixel噪声测试
# marker_pixels_cal = marker_pixels_cal + 5.0*np.random.randn(N,2)).astype(int)

print("uav_pose:\n",uav_pose)
print(depth_im.shape)
marker_depth = depth_im[tuple(marker_pixels_cal.T)]
# marker_depth = np.loadtxt(os.path.join(base_dir,"%d_depth_im.txt"%(n))).reshape(N)
print("marker_depth = \n", marker_depth)

print("K = ", K)
print("marker_pixels = \n", marker_pixels)

uv_const = np.ones(N)
marker_uvs = np.insert(marker_pixels[0:N,:],2,uv_const,axis=1)
print("marker_uvs = \n", marker_uvs)


## 计算markers的相机坐标
marker_depth = marker_depth[np.newaxis,:][:,0:N]
print(marker_depth.shape)
marker_depths = np.concatenate((marker_depth,marker_depth,marker_depth),axis = 0)
print("marker_depths = \n", marker_depths)
cam_cords = (marker_depths * (np.linalg.inv(K)@(marker_uvs.T))).T
cam_cords = np.concatenate((cam_cords[:,2][:,np.newaxis],-cam_cords[:,0][:,np.newaxis],-cam_cords[:,1][:,np.newaxis]),axis = 1)


print("marker_cords = \n", marker_cords)
print("cam_cords = \n", cam_cords)
print("err = \n", marker_cords - cam_cords)
ave_err = np.average(marker_cords - cam_cords, axis=0)
print("ave_err = \n",ave_err)



print(marker_cords)
index  = [0,N]

T, R, t = icp.SVD_ICP(marker_cords[index[0]:index[1],:], cam_cords[index[0]:index[1],:])
# T, R, t = ransac.pyRansac(marker_cords[index[0]:index[1],:], cam_cords[index[0]:index[1],:])
print("transform matrix is = \n", T)
t = t[:,np.newaxis]
print((R@marker_cords.T  - cam_cords.T + t)[:,index[0]:index[1]])
loss = np.linalg.norm(R@marker_cords.T + t - cam_cords.T, axis=0)
print("loss is " ,loss[index[0]:index[1]])


# np.save('data/cam_calib_V26/transform.npy',T)
print("offset_final = \n", np.linalg.inv(T@uav_pose))

##可视化
transform_cords = (R@(marker_cords.T)+t).T
# print("marker radius = ",R@np.array([0.07,0.0,0.0]).T)
fig = plt.figure(figsize=(12, 12))
ax3d = fig.add_subplot(111,projection="3d")  # 创建三维坐标
ax3d.plot(cam_cords[index[0]:index[1],0],cam_cords[index[0]:index[1],1],cam_cords[index[0]:index[1],2], marker='*', c="r", label = "cam_cords")
# ax3d.plot(marker_cords[:,0],marker_cords[:,1],marker_cords[:,2], marker='*', c="b", label = "marker_cords")
ax3d.plot(transform_cords[index[0]:index[1],0],transform_cords[index[0]:index[1],1],transform_cords[index[0]:index[1],2], marker='*', c="g", label = "transform_cords")
ax3d.set_xlabel('x', fontsize=10)
ax3d.set_ylabel('y', fontsize=10)
ax3d.set_zlabel('z', fontsize=10)
ax3d.set_title('ICP match', fontsize=10)

# ax = plt.gca()
# ax.set_aspect(1)
plt.legend()
plt.show()









