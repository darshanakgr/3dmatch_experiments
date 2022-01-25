import numpy as np
import open3d
import cv2


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


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
    pcd1 = open3d.read_point_cloud("data/3DMatch/rgbd_fragments/sun3d-hotel_sf-scan1/seq-01/cloud_bin_1.ply")
    pcd2 = open3d.read_point_cloud("data/3DMatch/rgbd_fragments/sun3d-hotel_sf-scan1/seq-01/cloud_bin_2.ply")
    # pcd3 = open3d.read_point_cloud("data/3DMatch/rgbd_fragments/sun3d-hotel_sf-scan1/seq-01/cloud_bin_3.ply")
    # pcd4 = open3d.read_point_cloud("data/3DMatch/rgbd_fragments/sun3d-hotel_sf-scan1/seq-01/cloud_bin_4.ply")

    pose1 = np.load("data/3DMatch/rgbd_fragments/sun3d-hotel_sf-scan1/seq-01/cloud_bin_1.pose.npy")
    pose2 = np.load("data/3DMatch/rgbd_fragments/sun3d-hotel_sf-scan1/seq-01/cloud_bin_2.pose.npy")
    # pose3 = np.load("data/3DMatch/rgbd_fragments/sun3d-hotel_sf-scan1/seq-01/cloud_bin_3.pose.npy")
    # pose4 = np.load("data/3DMatch/rgbd_fragments/sun3d-hotel_sf-scan1/seq-01/cloud_bin_4.pose.npy")

    pcd1 = open3d.voxel_down_sample(pcd1, 0.1)
    pcd2 = open3d.voxel_down_sample(pcd2, 0.1)
    # pcd3 = open3d.voxel_down_sample(pcd3, 0.1)
    # pcd4 = open3d.voxel_down_sample(pcd4, 0.1)


    pcd1.transform(pose1)
    pcd2.transform(pose2)

    pcd1_outliers = find_outliers(pcd1, threshold=10)
    pcd2_outliers = find_outliers(pcd2, threshold=10)

    # pcd3.transform(pose3)
    # pcd4.transform(pose4)

    # open3d.draw_geometries([pcd1, pcd2])

    match_inds = []
    bf_matcher = cv2.BFMatcher(cv2.NORM_L2)

    query_points = np.asarray(pcd1.points).astype(np.float32)
    train_points = np.asarray(pcd2.points).astype(np.float32)

    match = bf_matcher.match(query_points, train_points)

    for match_val in match:
        if match_val.distance < 0.1:
            if (match_val.queryIdx not in pcd1_outliers) and (match_val.trainIdx not in pcd2_outliers):
                match_inds.append([match_val.queryIdx, match_val.trainIdx])

    match_inds = np.array(match_inds)
    query_points_matched = query_points[match_inds[:, 0]]
    train_points_matched = train_points[match_inds[:, 1]]
    points = np.concatenate((query_points_matched, train_points_matched), axis=0)
    lines = [[i, i + len(train_points_matched)] for i in range(len(match_inds))]
    colors = [[1, 0, 0] for i in range(len(match_inds))]
    line_set = open3d.geometry.LineSet()
    line_set.points = open3d.utility.Vector3dVector(points)
    line_set.lines = open3d.utility.Vector2iVector(lines)
    line_set.colors = open3d.utility.Vector3dVector(colors)

    pcd1.paint_uniform_color([1, 0.706, 0])
    pcd2.paint_uniform_color([0, 0.651, 0.929])
    open3d.visualization.draw_geometries([pcd1, pcd2, line_set])


    # print(len(match_inds) / np.mean([query_points.shape[0], train_points.shape[0]]))


if __name__ == '__main__':
    main()
