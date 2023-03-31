import numpy as np
import matplotlib.pyplot as plt
import pylab
import os
from scipy.spatial.transform import Rotation as Ro
import  imageio

base_dir = os.path.join("data2\monosdf\l435\Video","test_all")
savePath = os.path.join("data2\monosdf\l435\Video","test_all")
rgb_list = []
for i in range(36):
    rgb = imageio.imread(os.path.join(base_dir,f"rendering_100_{i}.png"))
    rgb_list.append(rgb)


writer = imageio.get_writer(f'{savePath}/rgb_video.mp4', fps=5)
for i in range(len(rgb_list)):
    writer.append_data(rgb_list[i])
writer.close()
