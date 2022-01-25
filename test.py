import pickle

import open3d
import os
import glob
import numpy as np
# import multiprocessing as mp
from multiprocessing import Process, Array, Manager


def make_point_cloud(pts):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pts)
    return pcd


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def remove_common_points(point_cloud, threshold=10):
    open3d.geometry.estimate_normals(
        point_cloud, search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30)
    )

    pcd_tree = open3d.geometry.KDTreeFlann(point_cloud)
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)

    outliers = []
    for i in range(points.shape[0]):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point_cloud.points[i], 30)
        v = np.std([angle_between(normals[i], normals[j]) for j in idx])
        if v < np.radians(threshold):
            outliers.append(i)
    return np.asarray(outliers)
    # points = np.delete(points, outliers, axis=0)
    #
    # return make_point_cloud(points)


def check_outlier(process_id, points_split, points, global_outliers, threshold):
    point_cloud = make_point_cloud(points)

    open3d.geometry.estimate_normals(
        point_cloud, search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30)
    )

    normals = np.asarray(point_cloud.normals)
    pcd_tree = open3d.geometry.KDTreeFlann(point_cloud)

    outliers = []
    for i in points_split:
        [k, idx, _] = pcd_tree.search_knn_vector_3d(points[i], 30)
        v = np.std([angle_between(normals[i], normals[j]) for j in idx])
        if v < np.radians(threshold):
            outliers.append(i)

    global_outliers[process_id] = outliers


def remove_redundant_points(point_cloud, threshold=10):
    manager = Manager()
    global_outliers = manager.dict()

    points = np.asarray(point_cloud.points)

    points_splits = np.split(
        np.arange(points.shape[0]),
        [i for i in range(points.shape[0] // 4, points.shape[0] - points.shape[0] % 4, points.shape[0] // 4)]
    )

    processes = []

    for i, points_split in enumerate(points_splits):
        p = Process(target=check_outlier, args=(i, points_split, points, global_outliers, threshold))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    outliers = []
    for v in global_outliers.values():
        outliers += v
    # print(points.shape[0])
    points = np.delete(points, outliers, axis=0)
    # print(points.shape[0])
    return make_point_cloud(points)


def find_outliers(point_cloud, threshold=10):
    open3d.geometry.estimate_normals(
        point_cloud, search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30)
    )

    pcd_tree = open3d.geometry.KDTreeFlann(point_cloud)
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)

    outliers = []
    for i in range(points.shape[0]):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point_cloud.points[i], 30)
        v = np.std([angle_between(normals[i], normals[j]) for j in idx])
        if v < np.radians(threshold):
            outliers.append(i)

    return np.asarray(outliers)


def main():
    # keypoints_file = pickle.load(open("data/Dataset/3DMatch_train_0.050_keypts.pkl", "rb"))
    # points_file = pickle.load(open("data/Dataset/3DMatch_train_0.050_points.pkl", "rb"))
    #
    # for file_ids in keypoints_file.keys():
    #     anc_id, pos_id = file_ids.split("@")
    #     anc_points = np.asarray(points_file[anc_id])
    #     anc_pcd = make_point_cloud(anc_points)
    #     pos_points = np.asarray(points_file[pos_id])
    #     pos_pcd = make_point_cloud(pos_points)
    #     anc_pcd.paint_uniform_color([1, 0.706, 0])
    #     pos_pcd.paint_uniform_color([0, 0.651, 0.929])
    #     open3d.draw_geometries([anc_pcd, pos_pcd])

    # files = glob.glob("data/3DMatch/rgbd_fragments/sun3d-hotel_sf-scan1/seq-01/*.ply")
    # for file in files:
    #     pcd = open3d.read_point_cloud(file)
    #     open3d.draw_geometries([pcd])

    # pcd1 = open3d.read_point_cloud("data/3DMatch/rgbd_fragments/sun3d-hotel_sf-scan1/seq-01/cloud_bin_1.ply")
    # pcd2 = open3d.read_point_cloud("data/3DMatch/rgbd_fragments/sun3d-hotel_sf-scan1/seq-01/cloud_bin_3.ply")

    # pose1 = np.load("data/3DMatch/rgbd_fragments/sun3d-hotel_sf-scan1/seq-01/cloud_bin_1.pose.npy")
    # pose2 = np.load("data/3DMatch/rgbd_fragments/sun3d-hotel_sf-scan1/seq-01/cloud_bin_3.pose.npy")

    # pcd1 = open3d.voxel_down_sample(pcd1, 0.1)
    # pcd2 = open3d.voxel_down_sample(pcd2, 0.1)

    # outliers_pcd1 = np.load("data/outliers/0.1/sun3d-hotel_sf-scan1/seq-01/cloud_bin_1.npy")
    # outliers_pcd2 = np.load("data/outliers/0.1/sun3d-hotel_sf-scan1/seq-01/cloud_bin_2.npy")

    # outliers_pcd1m = find_outliers(pcd1, 10)
    # pcd3_points = np.delete(np.asarray(pcd1.points), outliers_pcd1, axis=0)
    # pcd3 = make_point_cloud(pcd3_points)
    #
    # pcd1.paint_uniform_color([1, 0.706, 0])
    # pcd3.paint_uniform_color([0, 0.651, 0.929])
    #
    # open3d.draw_geometries([pcd3, pcd1])
    # print(outliers_pcd1.shape)
    # print(outliers_pcd1m.shape)
    # print(np.intersect1d(outliers_pcd1, outliers_pcd1m).shape)
    # pcd1_points = np.asarray(pcd1.points)
    # len_before = pcd1_points.shape[0]
    # pcd1_points = np.delete(pcd1_points, outliers_pcd1, axis=0)
    # len_after = pcd1_points.shape[0]
    # new_pcd1 = make_point_cloud(pcd1_points)
    # print(len_before, len_after)
    # open3d.draw_geometries([pcd1])
    # open3d.draw_geometries([new_pcd1])
    # x2 = make_point_cloud(np.delete(np.asarray(pcd2.points), outliers_pcd2, axis=0))
    #
    # open3d.draw_geometries([pcd1])
    # open3d.draw_geometries([pcd2])
    #
    # open3d.draw_geometries([x1])
    # open3d.draw_geometries([x2])

    # vis = open3d.visualization.Visualizer()
    # vis.create_window()
    # vis.get_render_option().point_color_option = open3d.visualization.PointColorOption.Color
    # vis.get_render_option().point_size = 3.0
    # vis.add_geometry(pcd1)
    #
    # vis.capture_screen_image("file.jpg", do_render=True)
    # vis.run()
    # vis.destroy_window()

    # pcd1.paint_uniform_color([1, 0.706, 0])
    # pcd2.paint_uniform_color([0, 0.651, 0.929])

    # pcd1.transform(pose1)
    # pcd2.transform(pose2)

    # open3d.draw_geometries([pcd1, pcd2])

    # outlier = remove_redundant_points(pcd1, 10)
    # print(outlier)
    # open3d.draw_geometries([pcd1])
    # open3d.draw_geometries([pcd2])

    # x1 = remove_common_points(pcd1, 10)
    # x2 = remove_redundant_points(pcd1, 10)
    #
    # for i in range(1, 10):
    #     print(f"Threshold: {i}")
    #     x1 = remove_common_points(pcd1, i)
    #     # x2 = remove_common_points(pcd2, i)
    #
    #     # open3d.draw_geometries([pcd1])
    #     # open3d.draw_geometries([pcd2])
    #     open3d.draw_geometries([x1])
    x1 = open3d.read_point_cloud("data/samples/sample.pcd")
    x2 = open3d.read_point_cloud("data/samples/sample_pre.pcd")
    x3 = open3d.read_point_cloud("data/samples/sample_post.pcd")

    open3d.draw_geometries([x1])
    open3d.draw_geometries([x2])
    open3d.draw_geometries([x3])


if __name__ == '__main__':
    main()
