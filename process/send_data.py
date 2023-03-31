import threading
import requests
import numpy as np
import sys
import json
import time
import os

base_dir = "D:\Code\Python\cam_calib\data1"
index_dir = "obj_v10"

## 上传图片和位姿到服务器
def upload_picture(index):
    print(os.path.join(base_dir,"{:d}_pose.txt".format(index)))
    up_files = {'rgb': open(os.path.join(base_dir,index_dir,"{:d}_main.png".format(index)), 'rb'),
           'depth': open(os.path.join(base_dir,index_dir,"{:d}_depth.png".format(index)), 'rb'),
           'pose': open(os.path.join(base_dir,index_dir,"{:d}_pose.txt".format(index)), 'rb')}
    
    response = requests.post("http://10.15.198.112:7000/upload_picture/", files = up_files)
    # print(response.text)
    return response.text



## 发送数据
for i in range(0,30,1):
    upload_picture(i)
    time.sleep(0.5)
