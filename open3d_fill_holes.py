"""
此代码使用open3d，进行3D点云和深度图的转换
http://www.open3d.org/docs/release/python_example/geometry/point_cloud/index.html#point-cloud-to-depth-py
"""
import struct

import open3d as o3d
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
from insightface.app import FaceAnalysis
import os


# 聚类
# with o3d.utility.VerbosityContextManager(
#        o3d.utility.VerbosityLevel.Debug) as cm:
#    labels = np.array(
#        pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

# max_label = labels.max()
# print(f"point cloud has {max_label + 1} clusters")
# colors = plt.get_cmap("tab20")(labels / (max_label if max_label &gt; 0 else 1))
# colors[labels &lt; 0] = 0
# pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

# segment_plane
# plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
#                                          ransac_n=3,
#                                          num_iterations=1000)
# [a, b, c, d] = plane_model
# print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
#
# inlier_cloud = pcd.select_by_index(inliers)
# inlier_cloud.paint_uniform_color([1.0, 0, 0])
# outlier_cloud = pcd.select_by_index(inliers, invert=True)
# o3d.visualization.draw_geometries([outlier_cloud],
#                                   zoom=0.3412,
#                                   front=[0.4257, -0.2125, -0.8795],
#                                   lookat=[2.6172, 2.0475, 1.532],
#                                   up=[-0.0694, -0.9768, 0.2024])


def detect_face(rgb_face):
    """
    This function uses the FaceAnalysis library to detect faces in an RGB image.
    Args:
        rgb_face (numpy.ndarray): An RGB image represented as a numpy array.

    Returns:
        list: A list of dictionaries, where each dictionary represents a detected face. Each dictionary contains keys for bounding box coordinates, keypoints, and other attributes.
    """

    # 1. detect face and landmarks
    app = FaceAnalysis(allowed_modules=['detection'])  # enable detection model only
    app.prepare(ctx_id=0, det_size=(640, 640))

    faces = app.get(rgb_face)  # detection, recognition -> box, kps, gender, age

    # cv2.namedWindow('face_detection', 0)
    # cv2.imshow('face_detection', rimg)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # rimg = app.draw_on(rgb_face, faces)
    # cv2.imwrite("./face_detection.jpg", rimg)

    return faces


def get_rgb_from_bin(rgb_file, w=640, h=480):
    """
    从bin文件读取RGB图像，用于intel-realsense相机
    Args:
        rgb_file:
    Returns:

    """
    binfile = open(rgb_file, 'rb', encoding=None)  # 打开二进制文件
    size = os.path.getsize(rgb_file)  # 获得文件大小
    YUV = np.zeros(shape=(int(h * 1.5), w), dtype=np.uint8)  # h=720,w=640
    for i in range(size):
        data = binfile.read(1)  # 每次输出一个字节
        num = struct.unpack('B', data)
        # print(data)
        # print(num[0])
        row = int(i / w)
        col = int(i % w)
        YUV[row, col] = num[0]
    binfile.close()
    rgb_face = cv2.cvtColor(YUV, cv2.COLOR_YUV2BGR_NV21)
    # rgb_face = cv2.rotate(rgb_face, cv2.ROTATE_90_CLOCKWISE)

    return rgb_face


def get_depth_from_bin(bin_file, instrinsic):
    """
    read data.bin and get depth image with instrinsic,用于intel-realsense相机
    Args:
        bin_file:
        instrinsic:
    Returns:

    """
    width, height, fx, fy, cx, cy = instrinsic
    # 读取二进制数据
    with open(bin_file, 'rb') as file:
        data = file.read()

    # 解析二进制数据，每个深度值占用2个字节
    depth_map = np.frombuffer(data, dtype=np.uint16).reshape((height, width))
    # cv2.namedWindow('depth', 0)
    # cv2.imshow('depth', depth_map)
    # cv2.waitKey()

    # 应用内参进行坐标转换
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    Z = depth_map  # / 1000.0  # 将深度值从毫米转换为米
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy

    # matplotlib.use('svg')
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # X = np.arange(0, width, 1)
    # Y = np.arange(0, height, 1)
    # # Z = Z.ravel()
    # # Data for three-dimensional scattered points
    # ax.scatter3D(X, Y, Z(X, Y), c=Z.ravel(), cmap='Greens')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # plt.show()

    return Z


def get_intrinsic_from_txt(txt_file):
    if not os.path.exists(txt_file):
        print('No intrinsic file')
    f = open(txt_file, 'r')
    content = f.readlines()[0]
    all = content.strip().split()

    params = dict()
    for param in all:
        key, value = param.split(':')
        params[key] = value

    return params


def scp_face(pts, nose_3d, radius):
    """
    crop face with sphere cropping
    Args:
        pts: all pts of point cloud
        nose_3d: nose tip pose
        radius: radius of sphere

    Returns:

    """

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    x0, y0, z0 = nose_3d
    indx = []
    for i in range(0, pts.shape[0]):
        Y = y[i]  # pc2
        X = x[i]  # pc1
        Z = z[i]

        dis = (x0 - X) * (x0 - X) + (y0 - Y) * (y0 - Y) + (z0 - Z) * (z0 - Z)
        if dis < radius * radius:
            indx.append(i)

    # visualize cropped point cloud
    # points = np.zeros((len(indx), 3), dtype=np.float32)
    cropped_pts = pts[indx, :]

    return cropped_pts


if __name__ == '__main__':

    # read rgb image
    # data_dir = r'testdata/record/612729198904052114-20240110142341'
    data_dir = r'/home/xt/zk/XJTU_DATA_2023_part1-4/record20231224083748/11309581-20231224080915'

    rgb_file = os.path.join(data_dir, 'rgb.bin')
    set_file = os.path.join(data_dir, 'set.txt')
    data_file = os.path.join(data_dir, 'data.bin')
    if rgb_file.endswith('.bin'):
        rgb_image = get_rgb_from_bin(rgb_file)
    else:
        rgb_image = cv2.imread(rgb_file, -1)

    # cv2.namedWindow('rgb_face', 0)
    # cv2.imshow('rgb_face', rgb_image)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join(data_dir, 'rgb.png'), rgb_image)

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
    depth_scale = 0.001

    # get depth image
    instrinsic = depth_width, depth_height, fx, fy, cx, cy
    depth_image = get_depth_from_bin(data_file, instrinsic)
    # cv2.imwrite(os.path.join(data_dir, 'depth.png'), np.array(depth_image, np.uint8))
    # plt.imshow(depth_image, cmap='gray')
    # plt.imsave(os.path.join(data_dir, 'depth.png'), depth_image, cmap='gray')
    plt.imsave('depth.png', depth_image, cmap='gray')

    w, h = depth_width, depth_height
    points = np.zeros((w * h, 3), dtype=np.float32)

    pcd_file = os.path.join(data_dir, 'pc.ply')
    if os.path.exists(pcd_file):  # read pcd
        pcd = o3d.io.read_point_cloud(pcd_file)
        pcd.estimate_normals()
        o3d.visualization.draw_geometries([pcd], window_name='original face', width=depth_width, height=depth_height)
    else:  # create pcd from depth image
        n = 0
        for y in range(h):
            for x in range(w):
                X = x
                Y = y
                Z = depth_image[y, x]
                points[n][2] = Z
                points[n][0] = Y
                points[n][1] = X
                n = n + 1

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
        # pcd.points = o3d.utility.Vector3dVector(points)

        # pcd.remove_duplicated_points()
        # pcd.remove_non_finite_points()
        # pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=0.2)

        # estimate normals
        # pcd.normals = o3d.utility.Vector3dVector(np.zeros(
        #     (1, 3)))  # invalidate existing normals
        pcd.estimate_normals()
        o3d.io.write_point_cloud("pcd.ply", pcd)
        # o3d.visualization.draw([pcd])

    # 1. detect face and landmarks in RGB image
    faces = detect_face(rgb_image)
    face = faces[0]
    landmarks = face['kps']
    nose_tip = landmarks[2]

    # transform pcd to depth
    pts = np.asarray(pcd.points)
    x = pts[:, 0]
    # x = np.array(x * fx)
    y = pts[:, 1]
    # y = np.array(y * fy)
    z = pts[:, 2]

    # visualize depth map and nose tip
    depth = np.reshape(z, (depth_height, depth_width))

    depth = depth_image
    min_z = np.min(z)
    max_z = np.max(z)
    depth_map = 255 * (depth - min_z) / (max_z - min_z)
    depth_map = depth_map.astype(np.uint8)

    nose_x = int(nose_tip[0])
    nose_y = int(nose_tip[1])
    # nose_z = depth[nose_y, nose_x]

    cv2.drawMarker(rgb_image, (nose_x, nose_y), (255, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=3)
    cv2.imshow('nose', rgb_image)
    cv2.waitKey()

    # # todo:将nose的2D坐标转换为3D坐标
    ind_nose = nose_x + nose_y * 400
    # ind_nose = (np.abs(z - nose_z)).argmin()
    z0 = z[ind_nose]  # -0.484
    x0 = x[ind_nose]  # -0.0605
    y0 = y[ind_nose]  # -0.093092

    z_nose = z[ind_nose - 100:ind_nose + 100]
    z_nose = z_nose[z_nose > 0]
    z0 = np.median(z_nose)
    if z0 == 0:
        print('error')

    # # 球切SCM——输入nose的3D坐标x0, yo, z0, 输出球内点云坐标
    # r = 45 / fx
    r = 1000

    nose_3d = np.array((x0, y0, z0))
    cropped_pts = scp_face(pts, nose_3d, r)
    face_pcd = o3d.geometry.PointCloud()
    face_pcd.points = o3d.utility.Vector3dVector(cropped_pts)

    # pcd.remove_duplicated_points()
    # pcd.remove_non_finite_points()
    # pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=0.2)
    # pcd.remove_radius_outlier(nb_points=5, radius=5)

    face_pcd.estimate_normals()  # 非常慢
    o3d.visualization.draw_geometries_with_vertex_selection([face_pcd], window_name='face_pcd', width=400, height=640)
    o3d.io.write_point_cloud("face_pcd.ply", face_pcd)

    # Create mesh from point cloud using Poisson Algorithm
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            face_pcd, depth=9)

    # remove low density vertices
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # visualize Poisson mesh
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    mesh.vertex.colors = o3d.core.Tensor([[0.0, 0.0, 0.0],
                                          [0.1, 0.1, 0.1],
                                          [0.2, 0.2, 0.2],
                                          [0.7, 0.7, 0.7]],
                                         o3d.core.float32,
                                         o3d.core.Device("CPU:0"))
    # mesh = mesh.to_legacy()
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh.to_legacy()], window_name='possion')

    # # get mesh
    # mesh = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh('alpha.ply'))
    # # mesh.paint_uniform_color(np.array([[0.5], [0.5], [0.5]]))

    # fill holes in mesh
    filled = mesh.fill_holes(hole_size=10000)

    # visualize filled holes
    filled.vertex.colors = o3d.core.Tensor([[0.0, 0.0, 0.0],
                                            [0.1, 0.1, 0.1],
                                            [0.2, 0.2, 0.2],
                                            [0.7, 0.7, 0.7]],
                                           o3d.core.float32,
                                           o3d.core.Device("CPU:0"))
    filled = filled.to_legacy()
    # filled.compute_triangle_normals()
    filled.compute_vertex_normals()
    o3d.visualization.draw_geometries([filled], title='possion')

    # o3d.visualization.draw_geometries([pcd], window_name="pcd of filling holes")

    # resample points
    # num_points = filled.points
    samp_pcd = filled.sample_points_uniformly(30000, use_triangle_normal=False)
    # samp_pcd = filled.sample_points_poisson_disk(30000, use_triangle_normal=False)  # 效果差

    o3d.visualization.draw_geometries([samp_pcd], window_name="resampled pcd")