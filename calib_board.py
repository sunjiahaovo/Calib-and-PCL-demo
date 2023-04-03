import numpy as np
import os
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import icp
import os

base_dir = "data/calib_v1"

# -*- coding: utf-8 -*-
import cv2

# 查找棋盘格 角点
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

# 棋盘格参数
corners_vertical = 8   # 纵向角点个数;
corners_horizontal = 6  # 纵向角点个数;
pattern_size = (corners_vertical, corners_horizontal)


def find_corners_sb(img):
    """
    查找棋盘格角点函数 SB升级款
    :param img: 处理原图
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 查找棋盘格角点;
    ret, corners = cv2.findChessboardCornersSB(gray, pattern_size, cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
    # if ret:
    #     # 显示角点
    #     cv2.drawChessboardCorners(img, pattern_size, corners, ret)
    return ret, corners


def find_corners(img):
    """
    查找棋盘格角点函数
    :param img: 处理原图
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点;
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                             cv2.CALIB_CB_FAST_CHECK +
                                             cv2.CALIB_CB_FILTER_QUADS)
    if ret:
        # 精细查找角点
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # 显示角点
        cv2.drawChessboardCorners(img, pattern_size, corners2, ret)


def drawCorners(img_src,corners,reprojected_pixs):
    
    # 创建显示窗口
    cv2.namedWindow("img", 0)
    cv2.resizeWindow("img", 1280, 720)
    
    print(corners.shape,reprojected_pixs.shape)
    # print(corners-reprojected_pixs)
    # cv2.drawChessboardCorners(img_src, pattern_size, corners, True)
    cv2.drawChessboardCorners(img_src, pattern_size, reprojected_pixs.astype(np.float32), True)
    
    # 显示图片
    cv2.imshow("img", img_src)
    cv2.waitKey(0)

    # cv2.destroyAllWindows()
    
def drawPNP(corners,reprojected_pixs):
    corners = corners.reshape(48,2)
    reprojected_pixs = reprojected_pixs.reshape(48,2)
    fig = plt.figure(figsize=(12, 9))
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

    
    
def get_board2cam():
    # 循环读取标定图片
    for i in range(0, 1):
        file_path = os.path.join(base_dir,str(i)+'_main.png' )
        img_src = cv2.imread(file_path)

        if img_src is not None:
            # 执行查找角点算法
            ret, corners = find_corners_sb(img_src)
            # find_corners(img_src)

    # print(ret,corners.shape)
    image_points = copy.deepcopy(corners).reshape(48,2).astype(np.float64)
    # print(image_points,image_points.shape)
    
    obj_points = []
    obj_points_opt = []
    for i in range(6):
        for j in range(8):
            obj_points.append([(i+1)*0.115,(9-(j+1))*0.115,-0.007])
            obj_points_opt.append([0,-(i+1)*0.115,-(j+1)*0.115,1])
            # obj_points_opt.append([(i+1)*0.115,(j+1)*0.115,0,1])
    obj_points = np.array(obj_points).astype(np.float64)
    obj_points_opt = np.array(obj_points_opt).astype(np.float64)
    # print(obj_points,obj_points.shape)
    
    camera_matrix = np.array([[607.89, 0, 320.26],
                    [0, 607.93, 238.11],
                    [0, 0, 1]])  

    print(image_points)
    # print(obj_points.shape,image_points.shape,camera_matrix.shape)
    ok, rot, trans= cv2.solvePnP(objectPoints = obj_points, imagePoints = image_points, cameraMatrix = camera_matrix, distCoeffs = None)
    print(ok, rot, trans)
    
    # Convert rotation into a 3x3 Rotation Matrix
    # rot[:,0] = -rot[:,0]
    rot3x3, _ = cv2.Rodrigues(rot)
    
    T0 = np.zeros((4,4))
    T0[:3,:3]=rot3x3
    T0[:3,3]=trans[:,0]
    T0[3,3]=1
    print(T0)
    # # print((T0@(obj_points_opt.T)).T)
    # T1 = np.array([[0, 0, 1,0],
    #             [-1, 0, 0,0],
    #             [0, -1,0,0],
    #             [0,0,0,1]])   
    # # T2 = np.linalg.inv(T1)@T0@T1
    # T2 = T1@T0@np.linalg.inv(T1)
    # print((T2@(obj_points_opt.T)).T)
    
    
    # Reproject model points into image
    object_points_world = np.asmatrix(rot3x3) * np.asmatrix(obj_points.squeeze().T) + np.asmatrix(trans)
    # print("cam_ord = ", object_points_world.T)
    reprojected_h = camera_matrix * object_points_world
    reprojected   = (reprojected_h[0:2, :] / reprojected_h[2, :])
    reprojection_errors = corners.squeeze().T - reprojected

    reprojection_rms = np.sqrt(np.sum(np.array(reprojection_errors) ** 2) / np.product(reprojection_errors.shape))

    # Print the results
    print("Reprojection RMS Error: %.3f Pixels" % ( reprojection_rms))
    
    
    reprojected_pixs = copy.deepcopy(reprojected.T.A)[:,None,:].astype(np.float64)
    print(corners.shape,reprojected_pixs.shape)
    # print(corners)
    # print(reprojected_pixs)
    print(type(corners),type(reprojected_pixs))

    # drawCorners(img_src,corners,reprojected_pixs)
    return T0


def get_optitrack2board():
    
    index  = [0,20]
    
    board_points = np.loadtxt(os.path.join(base_dir,'board_points.txt'))
    markers_points = np.loadtxt(os.path.join(base_dir,'markers.txt'))
    # print(board_points,markers_points)
    board_points[:,0:2] = 0.115*board_points[:,0:2]
    board_points = np.concatenate((board_points[:,2][:,np.newaxis],-board_points[:,0][:,np.newaxis],-board_points[:,1][:,np.newaxis]),axis = 1)
    markers_points = np.concatenate((markers_points[:,0][:,np.newaxis],-markers_points[:,2][:,np.newaxis],markers_points[:,1][:,np.newaxis]),axis = 1)
    print("board_points = ", board_points)
    print("markers_points = ", markers_points)
    T, R, t = icp.SVD_ICP(markers_points, board_points)
    # T, R, t = ransac.pyRansac(markers_points[index[0]:index[1],:], board_points[index[0]:index[1],:])
    print("transform matrix is = \n", T)
    t = t[:,np.newaxis]
    print(R@markers_points.T + t - board_points.T)
    loss = np.linalg.norm(R@markers_points.T + t - board_points.T, axis=0)
    print("loss is " ,loss)
        
    ##可视化
    transform_cords = (R@(markers_points.T)+t).T
    # print("marker radius = ",R@np.array([0.07,0.0,0.0]).T)
    fig = plt.figure(figsize=(12, 12))
    ax3d = fig.add_subplot(111,projection="3d")  # 创建三维坐标
    ax3d.plot(board_points[index[0]:index[1],0],board_points[index[0]:index[1],1],board_points[index[0]:index[1],2], marker='*', c="r", label = "cam_cords")
    ax3d.plot(markers_points[:,0],markers_points[:,1],markers_points[:,2], marker='*', c="b", label = "marker_cords")
    ax3d.plot(transform_cords[index[0]:index[1],0],transform_cords[index[0]:index[1],1],transform_cords[index[0]:index[1],2], marker='*', c="g", label = "transform_cords")
    ax3d.set_xlabel('x', fontsize=10)
    ax3d.set_ylabel('y', fontsize=10)
    ax3d.set_zlabel('z', fontsize=10)
    ax3d.set_title('ICP match', fontsize=10)
    # ax3d.set_xlim(-1,1)

    # ax = plt.gca()
    # ax.set_aspect(1)
    plt.legend()
    plt.show()
    
    
    
    
    
    return T
    


if __name__ == '__main__':
    T_o2b = get_optitrack2board()
    Tt1 = np.array([[0, -1, 0,0],
                [0, 0, -1,0],
                [1, 0,0,0],
                [0,0,0,1]]) 
    
    T_b2c = get_board2cam()
    
    Tt2 = np.array([[0, 0, 1,0],
                    [-1, 0, 0,0],
                    [0, -1,0,0],
                    [0,0,0,1]])   
    
    print("T_o2b = ",T_o2b)
    print("T_b2c = ",T_b2c)
    T_o2c = Tt2@T_b2c@Tt1@T_o2b
    print("T_o2c = ",T_o2c)
    uav_pose = np.loadtxt(os.path.join(base_dir,"%d_pose.txt"%(0)),delimiter=",")
    print("uav_pose = ",uav_pose)
    print("offset_final = \n", np.linalg.inv(T_o2c@uav_pose))

