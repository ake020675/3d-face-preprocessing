"""
此代码用于学习3D人脸数据处理
2023.12.18
"""

import cv2
import numpy as np

from insightface.app import FaceAnalysis

import open3d as o3d


# def surfnorm_ocatave(depth):
#     """
#     implement surfnorm according to occave code
#     https://github.com/gnu-octave/octave/blob/773dbf2e1887ac771e0d24dd8dd55b42c82686bc/scripts/plot/draw/surfnorm.m#L104
#
#     Args:
#         depth:
#
#     Returns:
#
#     """
#     x = np.array(range(depth.shape[0]))
#     y = np.array(range(depth.shape[1]))
#     x, y = np.meshgrid(x, y)  # 3维网格坐标矩阵
#     z = depth
#
#     xx = [2 * x[:, 0] - x[:, 1], x, 2 * x[:, 2] - x[:, 1]]  # 列相减
#
#     # xx[0,:] = 2 * xx[0,:] - xx[1,:]
#     # xx[1,:] = xx
#     # xx[2, :] = 2 * xx[2, :] - xx[1,:]
#
#     xx = [2 * xx[0, :] - xx[1, :]; xx; 2 * xx[2, :] - xx[1, :]]  # 行相减
#     yy = [2 * y[:, 0] - y[:, 1], y, 2 * y[:, 2] - y[:, 1]]
#     yy = [2 * yy[0, :] - yy[1, :]; yy; 2 * yy[2, :] - yy[1, :]]
#     zz = [2 * z[:, 0] - z[:, 1], z, 2 * z[:, 2] - z[:, 1]]
#     zz = [2 * zz[0, :] - zz[1, :]; zz; 2 * zz(2,:] - zz[1, :]]
#
#     u.x = xx[:end - 1, : end - 1] - xx[1: , 1: ]
#     u.y = yy[:end - 1, : end - 1] - yy[1: , 1: ]
#     u.z = zz[:end - 1, : end - 1] - zz[1: , 1: ]
#     v.x = xx[:end - 1, 1: ] - xx[1: , : end - 1]
#     v.y = yy[:end - 1, 1: ] - yy[1: , : end - 1]
#     v.z = zz[:end - 1, 1: ] - zz[1: , : end - 1]
#
#     c = cross([u.x(:), u.y(:), u.z(:)], [v.x(:), v.y(:), v.z(:)]);
#     w.x = reshape(c(:, 1), size(u.x));
#     w.y = reshape(c(:, 2), size(u.y));
#     w.z = reshape(c(:, 3), size(u.z));
#
#     ## Create normal vectors as mesh vectices from normals at mesh centers
#     nx = (w.x(1:end-1, 1:end-1) + w.x(1:end - 1, 2: end) +
#     w.x(2: end, 1: end - 1) + w.x(2: end, 2: end)) / 4;
#     ny = (w.y(1:end-1, 1:end-1) + w.y(1:end - 1, 2: end) +
#         w.y(2: end, 1: end - 1) + w.y(2: end, 2: end)) / 4;
#     nz = (w.z(1:end-1, 1:end-1) + w.z(1:end - 1, 2: end) +
#         w.z(2: end, 1: end - 1) + w.z(2: end, 2: end)) / 4;
#
#     return [nx, ny, nz]


def o3d_fill_holes(depth):
    """
    todo: fill holes in point cloud
    :param depth: (h, w) of uint16, the unit of depth is mm
    """

    # # transform depth to pcd
    # # params
    # camera_factor = 1000
    # camera_cx = 323.63672
    # camera_cy = 212.21968
    # camera_fx = 521.97
    # camera_fy = 521.97

    camera_factor = 1
    camera_cx = 0
    camera_cy = 0
    camera_fx = 1
    camera_fy = 1

    # create pcd from depth image
    h, w = depth.shape[:2]
    points = np.zeros((w * h, 3), dtype=np.float32)
    n = 0
    for i in range(h):
        for j in range(w):
            points[n][2] = depth[i, j]/camera_factor

            # if points[n][2] < 0.4 or points[n][2] > 0.9:  # remove background
            #     points[n][2] = 0

            points[n][1] = (i-camera_cy)  # *points[n][2]/camera_fy
            points[n][0] = (j-camera_cx)  # *points[n][2]/camera_fx
            n = n + 1

    points = points[points[:, 2] > 0]  # remove background points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
    pcd.remove_duplicated_points()
    pcd.remove_non_finite_points()
    pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.2)

    # estimate normals
    pcd.normals = o3d.utility.Vector3dVector(np.zeros(
        (1, 3)))  # invalidate existing normals
    pcd.estimate_normals()
    o3d.io.write_point_cloud("pcd.ply", pcd)
    o3d.visualization.draw([pcd],  title="face pcd")

    # create mesh from pcd
    # # 1.alpha mesh
    # tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    # # for alpha in np.logspace(np.log10(2.5), np.log10(0.1), num=2):  # with different alpha values
    # #     print(f"alpha={alpha:.3f}")
    # #     mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
    # #         pcd, alpha, tetra_mesh, pt_map)
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
    #     pcd, 2.5, tetra_mesh, pt_map)  # alpha=25

    # 2. ball pivoting
    # # 滚球半径的估计
    # distances = pcd.compute_nearest_neighbor_distance()
    # avg_dist = np.mean(distances)
    # radius = 1.5 * avg_dist
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #     pcd,
    #     o3d.utility.DoubleVector([radius, radius * 2]))
    # print(mesh.get_surface_area())

    # radii = [0.005, 0.01, 0.02, 0.04]
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #     pcd, o3d.utility.DoubleVector(radii))

    # 3. poisson mesh
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    # o3d.visualization.draw([mesh])

    # remove low density vertices
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # visualize filled holes
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    mesh.vertex.colors = o3d.core.Tensor([[0.0, 0.0, 0.0],
                                            [0.1, 0.1, 0.1],
                                            [0.2, 0.2, 0.2],
                                            [0.3, 0.3, 0.3]],
                                           o3d.core.float32,
                                           o3d.core.Device("CPU:0"))
    mesh.compute_vertex_normals()
    mesh = mesh.to_legacy()
    o3d.visualization.draw_geometries([mesh], window_name='poisson', mesh_show_back_face=True)
    o3d.io.write_triangle_mesh("poisson.ply", mesh)

    # fill holes in mesh
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    filled = mesh.fill_holes(hole_size=100)

    # visualize filled holes
    filled.vertex.colors = o3d.core.Tensor([[0.0, 0.0, 0.0],
                                            [0.1, 0.1, 0.1],
                                            [0.2, 0.2, 0.2],
                                            [0.3, 0.3, 0.3]],
                                           o3d.core.float32,
                                           o3d.core.Device("CPU:0"))
    filled = filled.to_legacy()
    filled.compute_vertex_normals()
    filled.compute_triangle_normals()

    o3d.io.write_triangle_mesh("filled.ply", filled)
    # o3d.visualization.draw([{'name': 'filled', 'geometry': filled}])

    # mesh to point cloud
    pcd = o3d.geometry.PointCloud()
    pts = np.asarray(filled.vertices)
    pcd.points = o3d.utility.Vector3dVector(pts.reshape(-1, 3))

    pcd.normals = o3d.utility.Vector3dVector(np.zeros(
        (1, 3)))  # invalidate existing normals
    pcd.estimate_normals()

    # visualize of filled holes
    # pcd = filled.sample_points_uniformly(h*w, use_triangle_normal=False)
    # pcd = filled.sample_points_poisson_disk(h*w, use_triangle_normal=False)
    o3d.visualization.draw_geometries([pcd], window_name="pcd of filling holes")

    # reproj pcd to depth
    intrinsic = o3d.core.Tensor([[1, 0, 1], [0, 1, 1],
                                 [0, 0, 1]])
    pcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
    pcd.estimate_normals()
    o3d.io.write_point_cloud("pcd_filled.ply", pcd.to_legacy())

    depth_reproj = pcd.project_to_depth_image(width=w,
                                              height=h,
                                              intrinsics=intrinsic,
                                              depth_scale=1,
                                              depth_max=50.0)

    depth_filled = np.asarray(depth_reproj.to_legacy())  # depth->ointCloud->depth
    min_z = np.min(depth_filled)  # 0.4176
    max_z = np.max(depth_filled)  # 115.7643
    depth_filled = (depth_filled - min_z) / (max_z - min_z) * 255.0
    depth_filled = depth_filled.astype(np.uint8)
    cv2.namedWindow('depth_filled', 0)
    cv2.imshow('depth_filled', depth_filled)
    cv2.waitKey()
    cv2.imwrite('depth_filled.png', depth_filled)


def get_surface_normal_by_depth(depth, K=None):
    """
    https://stackoverflow.com/questions/53350391/surface-normal-calculation-from-depth-map-in-python
    depth: (h, w) of float, the unit of depth is meter
    K: (3, 3) of float, the depth camere's intrinsic
    """
    K = [[1, 0], [0, 1]] if K is None else K
    fx, fy = K[0][0], K[1][1]

    dz_dv, dz_du = np.gradient(depth)  # u, v mean the pixel coordinate in the image
    # u*depth = fx*x + cx --> du/dx = fx / depth
    du_dx = fx / depth  # x is xyz of camera coordinate
    dv_dy = fy / depth

    dz_dx = dz_du * du_dx
    dz_dy = dz_dv * dv_dy
    # cross-product (1,0,dz_dx)X(0,1,dz_dy) = (-dz_dx, -dz_dy, 1)
    normal_cross = np.dstack((-dz_dx, -dz_dy, np.ones_like(depth)))
    # normalize to unit vector
    normal_unit = normal_cross / np.linalg.norm(normal_cross, axis=2, keepdims=True)
    # set default normal to [0, 0, 1]
    normal_unit[~np.isfinite(normal_unit).all(2)] = [0, 0, 1]
    return normal_unit


def get_normal_map_by_point_cloud(depth, K):
    height, width = depth.shape

    def normalization(data):
        mo_chang = np.sqrt(
            np.multiply(data[:, :, 0], data[:, :, 0])
            + np.multiply(data[:, :, 1], data[:, :, 1])
            + np.multiply(data[:, :, 2], data[:, :, 2])
        )
        mo_chang = np.dstack((mo_chang, mo_chang, mo_chang))
        return data / mo_chang

    x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x = x.reshape([-1])
    y = y.reshape([-1])
    xyz = np.vstack((x, y, np.ones_like(x)))
    pts_3d = np.dot(np.linalg.inv(K), xyz * depth.reshape([-1]))
    pts_3d_world = pts_3d.reshape((3, height, width))
    f = (
            pts_3d_world[:, 1: height - 1, 2:width]
            - pts_3d_world[:, 1: height - 1, 1: width - 1]
    )
    t = (
            pts_3d_world[:, 2:height, 1: width - 1]
            - pts_3d_world[:, 1: height - 1, 1: width - 1]
    )
    normal_map = np.cross(f, t, axisa=0, axisb=0)
    normal_map = normalization(normal_map)
    return normal_map


# # test surface norm
# normal1 = get_surface_normal_by_depth(depth, K)    #  spend time: 60ms
# normal2 = get_normal_map_by_point_cloud(depth, K)  #  spend time: 90ms
#
# vis_normal = lambda normal: np.uint8((normal + 1) / 2 * 255)[..., ::-1]
# cv2.imwrite("normal1.png", vis_normal(normal1))
# cv2.imwrite("normal2.png", vis_normal(normal2))


def detect_face(rgb, input_size=(640, 640)):
    """
    This function uses the FaceAnalysis library to detect faces in an RGB image.
    Args:
        rgb (numpy.ndarray): An RGB image represented as a numpy array.

    Returns:
        list: A list of dictionaries, where each dictionary represents a detected face. Each dictionary contains keys for bounding box coordinates, keypoints, and other attributes.
    """

    # 1. detect face and landmarks
    app = FaceAnalysis(allowed_modules=['detection'])  # enable detection model only
    app.prepare(ctx_id=0, det_size=input_size)

    # app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # app.prepare(ctx_id=0, det_size=(640, 640))
    # img = ins_get_image('t1')  # cv2.imread(to_rgb\add dim)

    faces = app.get(rgb_face)  # detection, recognition -> box, kps, gender, age
    rimg = app.draw_on(rgb_face, faces)

    # cv2.namedWindow('face_detection', 0)
    # cv2.imshow('face_detection', rimg)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # cv2.imwrite("./output.jpg", rimg)

    return faces


if __name__ == '__main__':

    # read depth and rgb image
    rgb_file = r'Lock3D/a001.png'
    depth_file = r'Lock3D/a001.dat'
    rgb_face = cv2.imread(rgb_file)
    depth_face = np.loadtxt(depth_file)

    # rgb_file = r'/home/xt/zk/data/XJTU_DATA_2023_part1-4/record20231224083748/11309581-20231224080915/rgb.png'
    # depth_file = r'/home/xt/zk/data/XJTU_DATA_2023_part1-4/record20231224083748/11309581-20231224080915/data.bin'
    # rgb_face = cv2.imread(rgb_file, 1)
    # rgb_face = rgb_face[:400, :, :]
    # # 读取二进制数据
    # with open(depth_file, 'rb') as file:
    #     data = file.read()
    # depth_face = np.frombuffer(data, dtype=np.uint16).reshape((400, 640))

    # 1. detect face and landmarks
    nose_tip = [0, 0]
    dat_nose_tip = [0, 0]

    faces = detect_face(rgb_face)
    face = faces[0]
    landmarks = face['kps']
    nose_tip = landmarks[2]
    nose_tip[0] = round(nose_tip[0])
    nose_tip[1] = round(nose_tip[1])
    nose_tip = nose_tip.astype(np.int32)

    if len(faces) != 1:
        print("multi face detected")

    r = 90
    roi_rgb = rgb_face[nose_tip[1] - r:nose_tip[1] + r,
          nose_tip[0] - r:nose_tip[0] + r]

    # debug
    cv2.namedWindow('roi_rgb', 0)
    cv2.imshow('roi_rgb', roi_rgb)
    cv2.waitKey()
    cv2.imwrite('roi_rgb.png', roi_rgb)

    # 2.crop face in depth image
    dat_nose_tip[0] = round(nose_tip[0]/3.75 - 20)
    dat_nose_tip[1] = round(nose_tip[1]/3 + 66 - 30)
    roi = depth_face[dat_nose_tip[1] - r:dat_nose_tip[1] + r,
                     dat_nose_tip[0] - r:dat_nose_tip[0] + r]
    resize = 360
    roi_face = cv2.resize(roi, (resize, resize))

    # center of nose tip
    x0 = int(resize / 2)
    y0 = int(resize / 2)
    roi_nose = roi_face[x0 - 11:x0 + 10, y0 - 11:y0 + 10]
    roi_nose = roi_nose[roi_nose > 0]
    z0 = np.median(roi_nose)  # z0 = 578.9344  # for debug

    lenth_resize = np.array(range(0, resize * resize))
    pc1 = np.floor(lenth_resize / resize) + 1  # dat水平索引
    pc2 = lenth_resize % resize + 1  # dat垂直索引
    pc3 = np.ravel(roi_face)  # face roi 深度

    pc = np.zeros((3, resize * resize))
    pc[0, :] = pc1
    pc[1, :] = pc2
    pc[2, :] = pc3

    # 球切SCM
    r = 100
    for i in range(0, resize * resize):
        y = pc2[i]  # pc2
        x = pc1[i]  # pc1
        z = pc3[i]

        dis = (x0 - x) * (x0 - x) + (y0 - y) * (y0 - y) + (z0 - z) * (z0 - z)
        if dis > r * r:
            pc3[i] = 0
            pc[2, i] = 0

    locs_nonzero = np.where(pc3 > 0)
    pc3 = pc3[locs_nonzero]
    pc_face = pc[:, locs_nonzero]
    pc_face = np.squeeze(pc_face, axis=1)
    # pc_face = pc[:, pc3 > 0]

    # 切割后的人脸坐标索引
    # pc_face3 = pc_face[2, :]
    # pc_face3 = max(pc_face3) - pc_face3

    x = pc_face[0, :]
    y = pc_face[1, :]
    z = pc_face[2, :]
    z = max(z) - z

    max_x = max(x)  # 272
    min_x = min(x)  # 82
    max_y = max(y)  # 279
    min_y = min(y)  # 83

    size_x = int(max_x - min_x)  # 190
    size_y = int(max_y - min_y)  # 196

    # 点云转换为矩阵
    depth = np.zeros((size_y + 1, size_x + 1), np.float64)
    norm_z = np.zeros((size_y + 1, size_x + 1), np.float64)
    for i in range(len(z)):
        X = int(x[i] - min_x)
        Y = int(y[i] - min_y)
        depth[Y, X] = z[i]

    min_z = min(z[z > 0])  # 0.4176
    max_z = max(z)  # 115.7643
    print(f'min_z: {min_z}, maxz: {max_z}')

    # BY ZK
    # Normalize to [0, 255]
    depth_image = np.zeros((size_y + 1, size_x + 1), np.uint8)
    locs = np.where(depth > 0)
    # norm_z[depth < 0] = 0

    for i in range(len(locs[0])):
        x, y = (locs[0][i], locs[1][i])
        d = depth[x, y]
        depth_image[x, y] = round((d - min_z) / (max_z - min_z) * 255)
        norm_z[x, y] = (d - min(z)) / (max(z) - min(z))
        # depth[depth > 0] = np.array((depth[depth > 0] - min_z) / (max_z - min_z) * 255, np.uint8)
    depth_image = cv2.rotate(depth_image, cv2.ROTATE_90_CLOCKWISE)  # ROTATE_90_COUNTERCLOCKWISE
    # norm_z = cv2.rotate(norm_z, cv2.ROTATE_90_CLOCKWISE)

    # medfilter
    print("SCM done!")
    cropped = np.array(depth_image, np.uint8)
    cropped = cv2.medianBlur(depth_image, 3)

    # debug
    cv2.namedWindow('cropped', 0)
    cv2.imshow('cropped', cropped)
    cv2.waitKey()
    cv2.imwrite('cropped.png', depth)

    # 3.holes filling
    depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)  # ROTATE_90_COUNTERCLOCKWISE
    o3d_fill_holes(depth)

    # # 3.1 detect holes
    # # 最大连通域检测（前景）
    # mask = depth.copy()  # copy! very important!
    # mask[depth > 0] = 255  # 面具处理：背景区域数值为0，人脸区域数值255
    #
    # mask = np.array(mask, np.uint8)
    # labelImage = np.zeros_like(mask)
    # retval, labelImage, stats, centroids = cv2.connectedComponentsWithStats(mask, 4)
    # areas = stats[:, 4]
    # ind_maxArea = np.argmax(areas)  # 面积最大的连通域序号
    #
    # # 只保留mask中面积最大的连通域掩膜
    # mask_maxArea = np.zeros_like(mask)
    # mask_maxArea[labelImage == ind_maxArea] = 255
    #
    # mask = cv2.bitwise_and(mask, mask_maxArea)
    # # mask[labelImage == ind_maxArea] = 255
    # # mask[labelImage != ind_maxArea] = 0
    #
    # # debug
    # cv2.namedWindow('face mask', 0)
    # cv2.imshow('face mask', mask)
    # cv2.waitKey()
    #
    # depth[mask == 0] = 0  # 背景深度置为0
    #
    # locs_maxArea = np.where(mask_maxArea == 255)
    # # cv2.drawContours(mask_maxArea, [locs_maxArea], 0, 255, -1)
    # contours = list()
    # contours, _ = cv2.findContours(mask_maxArea, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # opencv4
    # for contour_index, contour in enumerate(contours):
    #     cv2.drawContours(mask_maxArea, contours, contour_index, 255, -1)
    #
    # # matlab-imfill
    # holes = mask.copy()
    # # im_floodfill = imfill(mask)
    # holes = cv2.absdiff(mask, mask_maxArea)  # 先找到孔洞位置
    #
    # # 3.2 fill holes by morph filter
    # dilate_holes = np.zeros_like(mask)
    # kernel = np.ones((3, 3), dtype=np.uint8)
    # cv2.dilate(holes, kernel, dilate_holes)  # 膨胀孔洞区域
    #
    # # debug
    # cv2.namedWindow('dilate_holes', 0)
    # cv2.imshow('dilate_holes', dilate_holes)
    # cv2.waitKey()
    #
    # dilate_holes_depth = depth.copy()
    # dilate_holes_depth[dilate_holes == 0] = 0  # 只取出孔洞周边小领域，进行imfill操作
    #
    # # debug
    # cv2.namedWindow('dilate_holes_depth', 0)
    # cv2.imshow('dilate_holes_depth', dilate_holes_depth)
    # cv2.waitKey()
    #
    # # from scipy_fill_hole import fill_image
    # # dilate_holes_depth, _ = fill_image(dilate_holes_depth)
    #
    #
    # # dilate_holes_depth = dilate_holes_depth.astype(np.float32)
    # # dilate_holes_depth = imfill(dilate_holes_depth, 'holes')
    #
    # # ordfilt2: rank=9时为最大值滤波器, 同dilate
    # # https://stackoverflow.com/questions/39772796/min-max-avg-filters-in-opencv2-4-13
    # # dilate_holes_depth = ordfilt2(dilate_holes_depth, 9, np.ones((3, 3)))
    # dilate_holes_depth = cv2.dilate(dilate_holes_depth, kernel)
    # dilate_holes_depth = cv2.medianBlur(dilate_holes_depth, 3)
    #
    #
    # # debug
    # cv2.imshow('dilate_holes_depth', dilate_holes_depth)
    # cv2.waitKey()
    # cv2.imwrite('dilate_holes_depth.png', dilate_holes_depth)
    #
    # # 将填充好的区域，补到深度图相应位置
    # depth[dilate_holes == 255] = dilate_holes_depth[dilate_holes == 255]
    #
    # # debug
    # # depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    # cv2.namedWindow('hole filled', 0)
    # cv2.imshow('hole filled', depth)
    # cv2.waitKey()
    #
    #
    # # 3.3 crop face point cloud again
    # # 找区域边界
    # x, y = np.where(mask == 255)
    # min_x = min(x)
    # max_x = max(x)
    # min_y = min(y)
    # max_y = max(y)
    # # 按边界切割mask
    # croped_mask = mask[min_x:max_x, min_y: max_y]  # 按区域边界，切割mask
    # # 按边界切割depth
    # croped_depth = depth[min_x:max_x, min_y: max_y]  # 按区域边界，切割mask
    #
    #
    # # debug
    # cv2.namedWindow('croped_depth', 0)
    # cv2.imshow('croped_depth', croped_depth)
    # cv2.waitKey()
    #
    # # todo: min_z error
    # # 4.get norm and project to 3 parts
    # # 归一化到[0, 255]
    # min_z = np.min(depth[depth > 0])  # 13
    # max_z = np.max(depth)  # 161
    # depth[depth > 0] = np.array((depth[depth > 0] - min_z) / (max_z - min_z) * 255, np.uint8)
    #
    # # 将深度图形状，转化成正方形
    # width = depth.shape[1]  # 196
    # height = depth.shape[0]  # 166
    # length = max(width, height)
    #
    # # 将原深度图，置于正方形的中心
    # size_depth = np.zeros((length, length))  # 196 * 196
    # size_depth[int((length - height) / 2): int((length - height) / 2) + height,
    #            int((length - width) / 2): (int((length - width) / 2) + width)] \
    #     = depth
    # depth = size_depth
    #
    # # 法向量投影 surfnorm
    # # [u, v, w] = surfnorm(uint8(depth))
    # normal = get_surface_normal_by_depth(depth)
    # vis_normal = lambda normal: np.uint8((normal + 1) / 2 * 255)[..., ::-1]
    # cv2.imwrite("normal.png", vis_normal(normal))

    # normal_x = np.array(((u + 1) * 128 - 1, np.uint8))  # rgb_R矩阵
    # normal_y = np.array(((v + 1) * 128 - 1, np.uint8))  # rgb_G矩阵
    # normal_z = np.array(((w + 1) * 128 - 1, np.uint8))  # rgb_B矩阵
    #
    # normal = np.zeros((depth.shape[0], depth.shape[1], 3))
    # normal[:, :, 0] = normal_x
    # normal[:, :, 1] = normal_y
    # normal[:, :, 2] = normal_z
