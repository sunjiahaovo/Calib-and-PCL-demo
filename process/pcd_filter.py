import open3d as o3d
import numpy as np

# 加载点云
pcd = o3d.io.read_point_cloud("Alexander_points.xyz")
# pcd = o3d.io.read_point_cloud("cabin-tsdf.xyz")
# o3d.visualization.draw_geometries([pcd])
# 半径滤波
num_points = 5  # 邻域球内的最少点数，低于该值的点为噪声点
radius = 0.5    # 邻域半径大小
# 执行半径滤波，返回滤波后的点云ror_pcd和对应的索引ind
ror_pcd, ind = pcd.remove_radius_outlier(num_points, radius)
ror_pcd.paint_uniform_color([0, 1, 0])
print("半径滤波后的点云：", ror_pcd)
# 提取噪声点云
ror_noise_pcd = pcd.select_by_index(ind,invert = True)
print("噪声点云：", ror_noise_pcd)
ror_noise_pcd.paint_uniform_color([1, 0, 0])

# np.savetxt("test_points_filter.csv",np.asarray(ror_pcd.points),delimiter=',',fmt='%.04f')
np.savetxt("Alexander_points_filter.csv",np.asarray(ror_pcd.points),delimiter=',',fmt='%.04f')
o3d.io.write_point_cloud("Alexander_points_filter.xyz",ror_pcd)

# 可视化滤波结果
o3d.visualization.draw_geometries([ror_pcd], window_name="半径滤波",
                                  width=800,  # 窗口宽度
                                  height=600)  # 窗口高度