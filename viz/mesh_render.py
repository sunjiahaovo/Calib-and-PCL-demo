# This demo code realize that first-person navigation in pointcloud.
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import imageio

odometry_file_path = "YOUR_ODOMETRY_FILE_PATH"
pcd_file_path = "data2/monosdf/l435/models/surface_70.ply"
 
# with open(odometry_file_path, 'r') as f:
#     content = f.readlines()
#     position_lis = content

mesh = o3d.io.read_triangle_mesh(pcd_file_path)

mesh.compute_vertex_normals()



print(mesh)

print('Vertices:')
print(np.asarray(mesh.vertices).shape)

print('Triangles:')
print(np.asarray(mesh.triangles))
print(np.asarray(mesh.triangle_normals))
triangles = np.asarray(mesh.triangles)
normals = np.asarray(mesh.triangle_normals).astype(np.float64)

# np.savetxt("triangles.txt",triangles)
# colors = np.full(np.asarray(mesh.vertices).shape,fill_value=[1,0,0])
# mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
print(triangles[triangles==308733])

colors = np.full(np.asarray(mesh.vertices).shape,fill_value=[0,0,0]).astype(np.float64)
print(colors.shape)

# for i in range(triangles.shape[0]):
#     for j in range(triangles.shape[1]):
#       colors[triangles[i,j]]+=normals[i]
# s

colors = np.loadtxt("colors.txt")
colors = colors[:, [1, 2, 0]]
colors[:,:2]= -colors[:,:2]
print(colors.shape)
colors_norm = np.linalg.norm(colors, axis=1)
colors[:,0]=colors[:,0]/colors_norm
colors[:,1]=colors[:,1]/colors_norm
colors[:,2]=colors[:,2]/colors_norm
print(colors.shape)
print(colors)
colors = (colors+1)/2
# mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
# o3d.visualization.draw_geometries([mesh])
o3d.io.write_triangle_mesh("color.ply",mesh) 

# mesh.paint_uniform_color([1, 0.706, 0])

# mesh.vertices.colors = np.full(np.asarray(mesh.vertices).shape,fill_value=[1,0,0])
# print(mesh.vertex)


# Real Scene
def get_pose_real(location,u,v):
    v = v/np.pi*180+180
    u = -u/np.pi*180 
    r = R.from_euler('ZYX', [v, 0, 0], degrees=True)
    pose = np.zeros((4,4))
    pose[0:3,0:3] = r.as_matrix()
    pose[0:3,3] = np.array([x,y,z])
    pose[3,3]=1
    # print("pose = ", pose)
    return pose


poses = []
for i in range(36):
    theta = i*10/180*np.pi
    x = -0.4*np.cos(theta)
    y = 0.3*np.sin(theta)
    z = -0.03
    object_center = np.array([0,0.,0.])
    dx,dy,dz =  object_center[0]-x,object_center[1]-y,object_center[2]-z
    v = np.arctan2(dy,dx) 
    pose = get_pose_real(np.array([x,y,z]),0,v)
    poses.append(pose.tolist())

poses = np.array(poses)


axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0.5,0.,0.])
vis = o3d.visualization.Visualizer()
vis.create_window(height=800, width=800)
vis.add_geometry(mesh)
# vis.add_geometry(axis_pcd)

writer = imageio.get_writer(f'tmp_rendering/mesh_normal_video.mp4', fps=5)

for idx in range(36):

    ctr = vis.get_view_control()
    pose = poses[idx]
    R = pose[:3,:3]
 
    origin_pos = pose[:3,3].tolist()
    front_vector = R[:, 0].tolist()
    up_vector = R[:, 2].tolist()
    # set up/lookat/front vector to vis
    ctr.set_front(front_vector)
    ctr.set_up(up_vector)
    ctr.set_lookat(origin_pos)
    init_param = ctr.convert_to_pinhole_camera_parameters()
    init_param.intrinsic.set_intrinsics(init_param.intrinsic.width, init_param.intrinsic.height, 600, 600, 400, 400)
    ctr.convert_from_pinhole_camera_parameters(init_param)

    print(pose)

#     ctr.set_constant_z_near(0)
#     ctr.set_constant_z_far(10)

#     # set the viewer's pose in the back of the first frame's pose
#     param = ctr.convert_to_pinhole_camera_parameters()
#     param.extrinsic = pose

#     ctr.convert_from_pinhole_camera_parameters(param)


    # because I want get first person view, so set zoom value with 0.001, if set 0, there will be nothing on screen.
    ctr.set_zoom(0.001)
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()

    os.makedirs(f'tmp_rendering', exist_ok=True)
    vis.capture_screen_image(f'tmp_rendering/{idx}.jpg')
    mesh_normal = imageio.imread(f'tmp_rendering/{idx}.jpg')
    writer.append_data(mesh_normal)
writer.close()
vis.run()
# vis.destroy_window()