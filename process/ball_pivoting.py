import open3d as o3d
import numpy as np

# 加载点云
pcd = o3d.io.read_point_cloud("cabin_err_v23_12.ply")
print(np.asarray(pcd.colors))

pcd.estimate_normals()

# BPA重建
radii = [0.005, 0.01, 0.02, 0.04]
bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
# 可视化重建结果
o3d.visualization.draw_geometries([bpa_mesh], window_name="点云重建",
                                  width=800,  
                                  height=600, 
                                  mesh_show_back_face=True)