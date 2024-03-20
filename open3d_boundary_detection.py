"""
https://www.open3d.org/docs/release/tutorial/t_geometry/pointcloud.html#Boundary-detection
"""


import os
from copy import deepcopy

from matplotlib import pyplot as plt
from sklearn import cluster

from open3d_fill_holes import get_rgb_from_bin, detect_face, get_intrinsic_from_txt, scp_face
import open3d as o3d
import cv2
import numpy as np


if __name__ == '__main__':

    # read rgb image
    data_dir = r'testdata/record/612729198904052114-20240110142341'
    rgb_file = os.path.join(data_dir, 'rgb.jpg')
    set_file = os.path.join(data_dir, 'set.txt')
    if rgb_file.endswith('.bin'):
        rgb_face = get_rgb_from_bin(rgb_file, set_file)
    else:
        rgb_face = cv2.imread(rgb_file, -1)

    # read params: fx, fy, cx, cy
    params = get_intrinsic_from_txt(set_file)
    fx = float(params['3d_fx'])
    fy = float(params['3d_fy'])
    cx = float(params['3d_cx'])
    cy = float(params['clor_cy'])
    color_width = int(params['clor_width'])
    clor_height = int(params['clor_height'])
    depth_width = int(params['3d_width'])
    depth_height = int(params['3d_height'])

    # read pcd
    pcd_file = os.path.join(data_dir, 'pc.ply')
    pcd = o3d.io.read_point_cloud(pcd_file)
    pcd.estimate_normals()

    # 1. detect face and landmarks in RGB image
    faces = detect_face(rgb_face)
    face = faces[0]
    landmarks = face['kps']
    nose_tip = landmarks[2]  # 125, 379

    # transform pcd to depth
    pts = np.asarray(pcd.points)
    x = pts[:, 0]
    # x = np.array(x * fx)
    y = pts[:, 1]
    # y = np.array(y * fy)
    z = pts[:, 2]

    # visualize depth map and nose tip
    depth = np.reshape(z, (depth_height, depth_width))
    min_z = np.min(z)  # -5.057
    max_z = np.max(z)  # -0.0
    depth_map = 255 * (depth - min_z) / (max_z - min_z)
    depth_map = depth_map.astype(np.uint8)

    nose_x = int(nose_tip[0])  # 125
    nose_y = int(nose_tip[1])  # 379
    nose_z = depth[nose_y, nose_x] * (max_z - min_z) / 255 + min_z

    # 将nose的2D坐标转换为3D坐标
    ind_nose = nose_x + nose_y * 400  # 151725
    # ind_nose = (np.abs(z - nose_z)).argmin()
    z0 = z[ind_nose]
    x0 = x[ind_nose]
    y0 = y[ind_nose]
    print(x0, y0, z0)  # -0.0692716 -0.0612456 -0.353

    # # 球切SCM——输入nose的3D坐标x0, yo, z0, 输出球内点云坐标
    r = 45 / fx
    nose_3d = np.array((x0, y0, z0))
    cropped_pts = scp_face(pts, nose_3d, r)
    face_pcd = o3d.geometry.PointCloud()
    face_pcd.points = o3d.utility.Vector3dVector(cropped_pts)

    # boundary points
    # pcd = o3d.t.io.read_point_cloud("face_pcd.ply")
    pcd = o3d.t.geometry.PointCloud.from_legacy(face_pcd)
    pcd.estimate_normals()
    boundarys, mask = pcd.compute_boundary_points(5, 120)
    # TODO: not good to get size of points.
    print(f"Detect {boundarys.point.positions.shape[0]} boundary points from {pcd.point.positions.shape[0]} points.")

    # remove outside points of pcd
    bd_pts = np.asarray(boundarys.to_legacy().points)
    cropped_pts = scp_face(bd_pts, nose_3d, r*0.6)
    boundarys = o3d.geometry.PointCloud()
    boundarys.points = o3d.utility.Vector3dVector(cropped_pts)

    # visualize boundary points inside pcd
    boundarys = o3d.t.geometry.PointCloud.from_legacy(boundarys)
    print(f"keep {boundarys.point.positions.shape[0]} boundary points inside pcd.")
    boundarys = boundarys.paint_uniform_color([1.0, 0.0, 0.0])
    pcd = pcd.paint_uniform_color([0.6, 0.6, 0.6])
    # o3d.visualization.draw_geometries([pcd.to_legacy(), boundarys.to_legacy()],
    #                                   window_name='boundary', width=400, height=640)
    o3d.io.write_point_cloud("boundary.ply", pcd.to_legacy())

    # todo: get bbox of boundary points and build mesh
    axis_aligned_bounding_box = pcd.get_axis_aligned_bounding_box()
    # axis_aligned_bounding_box.color = o3d.utility.Vector3dVector(np.array((1, 0, 0), np.float64))

    oriented_bounding_box = pcd.get_oriented_bounding_box()
    # axis_aligned_bounding_box.color = o3d.utility.Vector3dVector(np.array((0, 1, 0), np.float64))
    # oriented_bounding_box.color = (0, 1, 0)


    # todo: debug the failure of o3d clustering
    # get statistical measures
    dists = pcd.to_legacy().compute_nearest_neighbor_distance()
    dists = np.asarray(dists)
    mean_dis = np.mean(dists)
    thr_dis = mean_dis * 10

    # o3d cluster_dbscan
    boundarys = boundarys.to_legacy()
    labels = np.array(boundarys.cluster_dbscan(eps=0.0065, min_points=5, print_progress=True))

    # # kmeans clustering
    # n_clusters = 3  # 聚类簇数
    # points = np.array(boundarys.to_legacy().points)
    # kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=42)
    # kmeans.fit(points)
    # labels = kmeans.labels_

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")

    if max_label >= 0:
        # # 随机构建n+1种颜色，这里需要归一化
        colors = plt.get_cmap("tab20")
        tmp =labels / (max_label if max_label > 0 else 1)
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        boundarys.colors = o3d.utility.Vector3dVector(colors[:, :3])

        # 显示颜色设置
        # t_colors = np.random.randint(0, 255, size=(max_label+1, 3))
        # colors = t_colors.astype(np.float32) / 255.0
        # colors = colors[labels]

        color = [1.0, 0.0, 0.0]
        # visualize clusters
        # boundarys.colors = o3d.utility.Vector3dVector(color)
        o3d.visualization.draw_geometries([pcd.to_legacy(), boundarys],
                                          window_name="cluster",
                                          width=400,
                                          height=640)

    print('done')

