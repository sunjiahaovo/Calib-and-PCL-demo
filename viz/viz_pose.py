import numpy as np
import open3d as o3d
import os
import time
from scipy.spatial.transform import Rotation as Ro

## 画线
def get_lines(poses,color_flag = 0):

    N = poses.shape[0]
    points = np.array([poses[i,:3,3].tolist() for i in range(N)])
    lines = np.array([[i,i+1] for i in range(N-1)])
    if color_flag == 0:
        colors = np.array([[0, 0, 0] for j in range(len(lines))])
    elif color_flag == 1:
        colors = np.array([[1, 0, 0] for j in range(len(lines))])
    else:
        colors = np.array([[0, 1, 0] for j in range(len(lines))])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def get_tensor_from_camera(RT, Tquad=False):
    """
    Convert transformation matrix to quaternion and translation.

    """
    from mathutils import Matrix
    R, T = RT[:3, :3], RT[:3, 3]
    rot = Matrix(R)
    quad = rot.to_quaternion()
    if Tquad:
        tensor = np.concatenate([T, quad], 0)
    else:
        tensor = np.concatenate([quad, T], 0)
    return tensor

def create_camera_actor(i, color_flag=0, scale=0.005):
    cam_points = scale * np.array([
        [0,   0,   0],
        [-1,  -1, 1.5],
        [1,  -1, 1.5],
        [1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [0.5, 1, 1.5],
        [0, 1.2, 1.5]])

    cam_lines = np.array([[1, 2], [2, 3], [3, 4], [4, 1], [1, 3], [2, 4],
                          [1, 0], [0, 2], [3, 0], [0, 4], [5, 7], [7, 6]])
    points = []
    for cam_line in cam_lines:
        begin_points, end_points = cam_points[cam_line[0]
                                              ], cam_points[cam_line[1]]
        t_vals = np.linspace(0., 1., 100)
        begin_points, end_points
        point = begin_points[None, :] * \
            (1.-t_vals)[:, None] + end_points[None, :] * (t_vals)[:, None]
        points.append(point)
    points = np.concatenate(points)
    if color_flag == 0:
        color = (0.0, 0.0, 0.0)
    elif color_flag == 1:
        color = (1.0, .0, .0)
    else:
        color = (.0, 1.0, .0)

    camera_actor = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(points))
    camera_actor.paint_uniform_color(color)

    return camera_actor


def create_traj_actor(poses, color_flag=0, scale=0.005):
    N = poses.shape[0]
    points = []
    for i in range(N-1):
        begin_points, end_points = poses[i,:3,3], poses[i+1,:3,3]
        t_vals = np.linspace(0., 1., 100)
        point = begin_points[None, :] * \
            (1.-t_vals)[:, None] + end_points[None, :] * (t_vals)[:, None]
        points.append(point)
    points = np.concatenate(points)
    if color_flag == 0:
        color = (0.0, 0.0, 0.0)
    elif color_flag == 1:
        color = (1.0, .0, .0)
    else:
        color = (.0, 1.0, .0)

    traj_actor = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(points))
    traj_actor.paint_uniform_color(color)
    

    return traj_actor

base_dir = "data2/monosdf/replica/scan1"


vis = o3d.visualization.Visualizer()
vis.create_window(window_name="output", height=1080, width=1920)

mesh = o3d.io.read_triangle_mesh(os.path.join(base_dir,"surface_200.ply"))
mesh.compute_vertex_normals()
# # 打印网格信息
# print(mesh)
# # 打印网格定点信息
# print('Vertices:')
# print(np.asarray(mesh.vertices))
# # 打印网格的三角形
# print('Triangles:')
# print(np.asarray(mesh.triangles))

# vis.add_geometry(mesh)


### 加载位姿
est_poses = np.loadtxt(os.path.join(base_dir,str(67)+"_pose.txt")).reshape(100,4,4)
gt_poses = []
noise_poses = []
for i in range(100):
    gt_pose = np.loadtxt(os.path.join(base_dir,str(i)+"_gt.txt"))
    noise_pose = np.loadtxt(os.path.join(base_dir,str(i)+".txt"))
    gt_poses.append(gt_pose.tolist())
    noise_poses.append(noise_pose.tolist())
gt_poses = np.array(gt_poses)
noise_poses = np.array(noise_poses)

noise_loss_all_r = []
noise_loss_all_t = []
est_loss_all_r = []
est_loss_all_t = []
## 计算loss
for i in range(100):
    gt_r ,gt_t = gt_poses[i,:3,:3], gt_poses[i,:3,3]
    noise_r ,noise_t = noise_poses[i,:3,:3], noise_poses[i,:3,3]
    est_r ,est_t = est_poses[i,:3,:3], est_poses[i,:3,3]
    noise_loss_r = Ro.from_matrix(noise_r@np.linalg.inv(gt_r)).as_euler('ZYX', degrees=True)
    est_loss_r = Ro.from_matrix(est_r@np.linalg.inv(gt_r)).as_euler('ZYX', degrees=True)
    # print(noise_loss_r,est_loss_r)
    noise_loss_all_r.append(abs(noise_loss_r).tolist())
    est_loss_all_r.append(abs(est_loss_r).tolist())
    noise_loss_all_t.append(abs(gt_t-noise_t).tolist())
    est_loss_all_t.append(abs(gt_t-est_t).tolist())

noise_loss_all_r = np.array(noise_loss_all_r)
noise_loss_all_t = np.array(noise_loss_all_t)
est_loss_all_r = np.array(est_loss_all_r)
est_loss_all_t = np.array(est_loss_all_t)

noise_err_all_r = np.mean(np.linalg.norm(noise_loss_all_r,axis=1))
est_err_all_r = np.mean(np.linalg.norm(est_loss_all_r,axis=1))
print("r_err = ",noise_err_all_r,"------->",est_err_all_r)
noise_err_all_t = np.mean(np.linalg.norm(noise_loss_all_t,axis=1))
est_err_all_t = np.mean(np.linalg.norm(est_loss_all_t,axis=1))
print("t_err = ",noise_err_all_t,"------->",est_err_all_t)


## 画轨迹
# gt_lines = get_lines(gt_poses,color_flag = 0)
# noise_lines = get_lines(noise_poses,color_flag = 1)
# est_lines = get_lines(est_poses,color_flag = 2)
# vis.add_geometry(gt_lines)
# vis.add_geometry(noise_lines)
# vis.add_geometry(est_lines)
traj_scale = 0.5
gt_traj = create_traj_actor(gt_poses, 0, traj_scale)
vis.add_geometry(gt_traj)
noise_traj = create_traj_actor(noise_poses, 1, traj_scale)
# vis.add_geometry(noise_traj)
est_traj = create_traj_actor(est_poses, 2, traj_scale)
vis.add_geometry(est_traj)


cam_scale = 0.04
for i in range(0,100,20):

    gt_pose = gt_poses[i]
    gt_pose[:3,1:3] = - gt_pose[:3,1:3]
    cam_actor = create_camera_actor(i, 0, cam_scale)
    cam_actor.transform(gt_pose)
    vis.add_geometry(cam_actor)

    noise_pose = noise_poses[i]
    noise_pose[:3,1:3] = - noise_pose[:3,1:3]
    cam_actor = create_camera_actor(i, 1, cam_scale)
    cam_actor.transform(noise_pose)
    # vis.add_geometry(cam_actor)


    est_pose = est_poses[i]
    est_pose[:3,1:3] = - est_pose[:3,1:3]
    cam_actor = create_camera_actor(i, 2, cam_scale)
    cam_actor.transform(est_pose)
    vis.add_geometry(cam_actor)

    vis.poll_events()
    vis.update_renderer()

vis.get_render_option().point_size = 3
vis.run()


