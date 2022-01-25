import numpy as np
import open3d


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

    points = np.delete(points, outliers, axis=0)

    return make_point_cloud(points)

