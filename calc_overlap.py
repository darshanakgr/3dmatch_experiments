import os
from os.path import exists, join
import pickle
import numpy as np
import open3d
import cv2
import time
import outlier_filter as filter


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


class ThreeDMatch(object):
    """
    Given point cloud fragments and corresponding pose in '{root}'.
        1. Save the aligned point cloud pts in '{savepath}/3DMatch_{downsample}_points.pkl'
        2. Calculate the overlap ratio and save in '{savepath}/3DMatch_{downsample}_overlap.pkl'
        3. Save the ids of anchor keypoints and positive keypoints in '{savepath}/3DMatch_{downsample}_keypts.pkl'
    """

    def __init__(self, root, savepath, split, downsample):
        self.root = root
        self.savepath = savepath
        self.split = split
        self.downsample = downsample

        # dict: from id to pts.
        self.pts = {}

        # dict: from id_id to overlap_ratio
        self.overlap_ratio = {}
        # dict: from id_id to anc_keypts id & pos_keypts id
        self.keypts_pairs = {}

        with open(os.path.join(root, f'scene_list_{split}.txt')) as f:
            scene_list = f.readlines()
        self.ids_list = []
        self.scene_to_ids = {}
        for scene in scene_list:
            scene = scene.replace("\n", "")
            self.scene_to_ids[scene] = []
            for seq in sorted(os.listdir(os.path.join(self.root, scene))):
                if not seq.startswith('seq'):
                    continue
                scene_path = os.path.join(self.root, scene + f'/{seq}')
                ids = [scene + f"/{seq}/" + str(filename.split(".")[0]) for filename in os.listdir(scene_path) if
                       filename.endswith('ply')]
                ids = sorted(ids, key=lambda x: int(x.split("_")[-1]))
                self.ids_list += ids
                self.scene_to_ids[scene] += ids
                print(f"Scene {scene}, seq {seq}: num ply: {len(ids)}")
        print(f"Total {len(scene_list)} scenes, {len(self.ids_list)} point cloud fragments.")
        self.idpair_list = []
        self.load_all_ply(downsample)
        self.cal_overlap(downsample)

    def load_ply(self, data_dir, ind, downsample, aligned=True):
        pcd = open3d.read_point_cloud(join(data_dir, f'{ind}.ply'))
        pcd = open3d.voxel_down_sample(pcd, voxel_size=downsample)
        if aligned is True:
            matrix = np.load(join(data_dir, f'{ind}.pose.npy'))
            pcd.transform(matrix)

        pcd = filter.remove_common_points(pcd, 10)
        return pcd

    def load_all_ply(self, downsample):
        pts_filename = join(self.savepath, f'3DMatch_{self.split}_{downsample:.3f}_points.pkl')
        if exists(pts_filename):
            with open(pts_filename, 'rb') as file:
                self.pts = pickle.load(file)
            print(f"Load pts file from {self.savepath}")
            return
        self.pts = {}
        for i, anc_id in enumerate(self.ids_list):
            anc_pcd = self.load_ply(self.root, anc_id, downsample=downsample, aligned=True)
            points = np.array(anc_pcd.points)
            # print(len(points))
            self.pts[anc_id] = points
            print('processing ply: {:.1f}%'.format(100 * i / len(self.ids_list)))
        with open(pts_filename, 'wb') as file:
            pickle.dump(self.pts, file)

    def get_matching_indices(self, anc_pts, pos_pts, search_voxel_size, K=None):
        match_inds = []
        bf_matcher = cv2.BFMatcher(cv2.NORM_L2)
        match = bf_matcher.match(anc_pts, pos_pts)
        for match_val in match:
            if match_val.distance < search_voxel_size:
                match_inds.append([match_val.queryIdx, match_val.trainIdx])
        return np.array(match_inds)

    def cal_overlap(self, downsample):
        overlap_filename = join(self.savepath, f'3DMatch_{self.split}_{downsample:.3f}_overlap.pkl')
        keypts_filename = join(self.savepath, f'3DMatch_{self.split}_{downsample:.3f}_keypts.pkl')
        if exists(overlap_filename) and exists(keypts_filename):
            with open(overlap_filename, 'rb') as file:
                self.overlap_ratio = pickle.load(file)
                print(f"Reload overlap info from {overlap_filename}")
            with open(keypts_filename, 'rb') as file:
                self.keypts_pairs = pickle.load(file)
                print(f"Reload keypts info from {keypts_filename}")
            import pdb
            pdb.set_trace()
            return
        t0 = time.time()
        for scene, scene_ids in self.scene_to_ids.items():
            scene_overlap = {}
            print(f"Begin processing scene {scene}")
            for i in range(0, len(scene_ids)):
                anc_id = scene_ids[i]
                for j in range(i + 1, len(scene_ids)):
                    pos_id = scene_ids[j]
                    anc_pts = self.pts[anc_id].astype(np.float32)
                    pos_pts = self.pts[pos_id].astype(np.float32)

                    try:
                        matching_01 = self.get_matching_indices(anc_pts, pos_pts, self.downsample)
                    except BaseException as e:
                        print(f"Something wrong with get_matching_indices {e} for {anc_id}, {pos_id}")
                        matching_01 = np.array([])
                    overlap_ratio = len(matching_01) / len(anc_pts)
                    # matching_10 = self.get_matching_indices(pos_pts, anc_pts, self.downsample)
                    # overlap_ratio = max(len(matching_01) / len(anc_pts), len(matching_10) / len(pos_pts))

                    scene_overlap[f'{anc_id}@{pos_id}'] = overlap_ratio
                    if overlap_ratio > 0.30:
                        self.keypts_pairs[f'{anc_id}@{pos_id}'] = matching_01.astype(np.int32)
                        self.overlap_ratio[f'{anc_id}@{pos_id}'] = overlap_ratio
                        print(f'\t {anc_id}, {pos_id} overlap ratio: {overlap_ratio}')
                print('processing {:s} ply: {:.1f}%'.format(scene, 100 * i / len(scene_ids)))
            print('Finish {:s}, Done in {:.1f}s'.format(scene, time.time() - t0))

        with open(overlap_filename, 'wb') as file:
            pickle.dump(self.overlap_ratio, file)
        with open(keypts_filename, 'wb') as file:
            pickle.dump(self.keypts_pairs, file)


if __name__ == '__main__':
    ThreeDMatch(root='data/3DMatch/rgbd_fragments',
                savepath='data/Dataset',
                split='train',
                downsample=0.05
                )
