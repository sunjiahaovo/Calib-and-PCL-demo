# coding=utf-8
import numpy as np
import cv2
import os
import argparse


base_dir = "data_for_pcl/jiahao_test5"
saved_out_dir = "data_for_pcl/jiahao_test5_imageselect"


fm_list = []
def variance_of_laplacian(image):
	'''
    计算图像的laplacian响应的方差值
    '''
	return cv2.Laplacian(image, cv2.CV_64F).var()

sum = 0
if __name__ == '__main__':
    # 设置参数
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--threshold", type=float, default=100.0, help="设置模糊阈值")
    args = vars(ap.parse_args())
    
    for i in np.arange(0,1585,1):
        # print(i)
        # 读取图片
        img = cv2.imread(os.path.join(base_dir,str(i)+"_main.png"))
        # 将图片转换为灰度图片
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 计算灰度图片的方差
        fm = variance_of_laplacian(gray)
        text = "Not Blurry"
        
        # 设置输出的文字
        if fm < args["threshold"]:
            text = "Blurry"
        else:
            fm_list.append([i,fm])
            sum += 1
        # print(i,fm,text)
    print("sum = ", sum)
    fm_arr = np.array(fm_list)
    fm_sort = fm_arr[np.lexsort(-fm_arr.T)]
    print(fm_sort)
    
    now_index = 0
    for i in range(fm_sort.shape[0]):
        origin_index = int(fm_sort[i,0])
        optitrack_pose = np.loadtxt(os.path.join(base_dir,str(origin_index)+"_optitrack_pose.txt"))
        t265_pose = np.loadtxt(os.path.join(base_dir,str(origin_index)+"_t265_pose.txt"))
        img = cv2.imread(os.path.join(base_dir,str(origin_index)+"_main.png"))
        depth = cv2.imread(os.path.join(base_dir,str(origin_index)+"_depth.png"), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        
        print(os.path.join(saved_out_dir,str(now_index)+"_optitrack_pose.txt"))
        np.savetxt(os.path.join(saved_out_dir,str(now_index)+"_optitrack_pose.txt"),optitrack_pose)
        np.savetxt(os.path.join(saved_out_dir,str(now_index)+"_t265_pose.txt"),t265_pose)
        cv2.imwrite(os.path.join(saved_out_dir,str(now_index)+"_main.png"),img)
        cv2.imwrite(os.path.join(saved_out_dir,str(now_index)+"_depth.png"),depth)
        
        now_index += 1
    print("finished !!!")