import os
import numpy as np

base_dir = "data2/obj_v5/"

for i in range(40,121,1):
    os.rename(os.path.join(base_dir,f'{i}_main.png'),os.path.join(base_dir,f'{i-20}_main.png'))
    os.rename(os.path.join(base_dir,f'{i}_depth.png'),os.path.join(base_dir,f'{i-20}_depth.png'))
    os.rename(os.path.join(base_dir,f'{i}_pose.txt'),os.path.join(base_dir,f'{i-20}_pose.txt'))
