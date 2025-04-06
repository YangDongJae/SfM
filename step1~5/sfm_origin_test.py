import os
import cv2
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN
from utils import read_directory, get_rgb
from bundle import BundleAdjustment
from bundle_gpu import BundleAdjustmentTorch

class Point3D:
    def __init__(self, pt):
        self.pt = pt
        self.idx_image = dict()
        self.pt2D = dict()

class SFM_TEST:
    def __init__(self):
        self.dir_path = ""
        self.num_images = 0
        self.world_index = 0
        self.path_images = []

        self.mat_color_images = []
        self.mat_gray_images = []
        self.camera_poses = []
        self.cam_positions = []

        self.keypoints = []
        self.descriptors = []
        self.points2D = []

        self.recon3Dpts = []
        self.rgb_values = []

        self.cam3Dpts = []
        self.cam3DRGB = []

        self.order_pair = dict()
        self.done_views = set()
        self.good_views = set()
        self.skip_views = set()

        self.intrinsic_data = np.array([[1698.873755, 0, 971.7497705],
                                        [0, 1698.8796645, 647.7488275],
                                        [0, 0, 1]], dtype=np.float32)

        self.intrinsic_test = np.array([[1291.6714967317,0.0000000000,644.2699986472],
                                        [0.0000000000,1287.4135827401,846.6170615545],
                                        [0, 0, 1]], dtype=np.float32)

        self.intrinsic = None
        self.distortion = np.zeros(5)

    def set_dir_path(self, num_images, dir_path):
        self.dir_path = dir_path
        self.num_images = num_images
        self.path_images = read_directory(dir_path)[:num_images]

    def load_and_extract_features(self, apply_clustering=True):
        if self.dir_path == "./data/":
            self.intrinsic = self.intrinsic_data
        else:
            self.intrinsic = self.intrinsic_test
            
        self.camera_poses = [np.eye(3, 4)] * self.num_images
        #-----------------------------DEBUGING : SIFT_MAXIMUM-----------------------------#
        sift = cv2.SIFT_create()
        #-----------------------------DEBUGING : SIFT_MAXIMUM-----------------------------#

        for idx, fname in enumerate(self.path_images):
            color = cv2.imread(os.path.join(self.dir_path, fname))
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            kp, desc = sift.detectAndCompute(gray, None)

            pts2D = np.array([kp[i].pt for i in range(len(kp))])

            #-----------------------------CLUSTERING-----------------------------#
            if apply_clustering and len(pts2D) > 0:
                db = DBSCAN(eps=150, min_samples=10).fit(pts2D)
                labels = db.labels_
                unique, counts = np.unique(labels[labels != -1], return_counts=True)
                if len(unique) > 0:
                    main_cluster = unique[np.argmax(counts)]
                    mask = labels == main_cluster
                    filtered_kp = [kp[i] for i in range(len(kp)) if mask[i]]
                    filtered_desc = desc[mask]
                    filtered_pts2D = pts2D[mask]
                    kp, desc, pts2D = filtered_kp, filtered_desc, filtered_pts2D

            self.mat_color_images.append(color)
            self.mat_gray_images.append(gray)
            self.keypoints.append(kp)
            self.descriptors.append(desc)
            self.points2D.append(pts2D)
            #-----------------------------CLUSTERING-----------------------------#
            print(f"{fname} has {len(kp)} clustered SIFT features")


    def match_features(self, i1, i2):
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        knn_matches = bf.knnMatch(self.descriptors[i1], self.descriptors[i2], k=2)
        good = [m[0] for m in knn_matches if len(m) == 2 and m[0].distance < 0.55 * m[1].distance]
        return good

    def compute_angle(self, pt1, pt2):
        delta = pt2 - pt1
        angle = np.degrees(np.arctan2(abs(delta[1]), abs(delta[0])))
        return angle

    def find_best_pair(self):
        match_scores = dict()
        for i in range(self.num_images):
            for j in range(i + 1, self.num_images):
                matches = self.match_features(i, j)
                if len(matches) < 80:
                    continue

                pts1 = np.float32([self.keypoints[i][m.queryIdx].pt for m in matches])
                pts2 = np.float32([self.keypoints[j][m.trainIdx].pt for m in matches])

                angles = [self.compute_angle(p1, p2) for p1, p2 in zip(pts1, pts2)]
                mask = [a <= 35 for a in angles]
                pts1 = pts1[mask]
                pts2 = pts2[mask]

                if len(pts1) < 80:
                    continue

                F, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_7POINT, 1.0, 0.99)
                if F is None or F.shape != (3, 3):
                    continue

                match_scores[len(pts1)] = (i, j)

        self.order_pair = dict(sorted(match_scores.items(), reverse=True))
        print("=== best pair found ===")

    def get_camera_pose(self, i1, i2, matches):
        pts1 = np.float32([self.keypoints[i1][m.queryIdx].pt for m in matches])
        pts2 = np.float32([self.keypoints[i2][m.trainIdx].pt for m in matches])
        
        #-----------------------------DEBUGING : RANSANC HYPER PARAMETER-----------------------------#
        E, mask = cv2.findEssentialMat(pts1, pts2, self.intrinsic, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        #-----------------------------DEBUGING : RANSANC HYPER PARAMETER-----------------------------#
        
        _, R, T, mask_pose = cv2.recoverPose(E, pts1, pts2, self.intrinsic)
        return np.hstack((np.eye(3), np.zeros((3, 1)))), np.hstack((R, T)), R, T

    def triangulate_points(self, i1, i2, P1, P2, matches):
        pts1 = np.float32([self.keypoints[i1][m.queryIdx].pt for m in matches])
        pts2 = np.float32([self.keypoints[i2][m.trainIdx].pt for m in matches])
        pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), self.intrinsic, self.distortion)
        pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), self.intrinsic, self.distortion)
        pts4d_h = cv2.triangulatePoints(P1, P2, pts1_norm, pts2_norm)
        pts4d = pts4d_h[:3] / pts4d_h[3]
        points3D = []
        for i, pt in enumerate(pts4d.T):
            p = Point3D(pt)
            p.idx_image[i1] = matches[i].queryIdx
            p.idx_image[i2] = matches[i].trainIdx
            p.pt2D[i1] = pts1[i]
            p.pt2D[i2] = pts2[i]
            points3D.append(p)
        return points3D, pts1

    def base_reconstruction(self):
        if not self.order_pair:
            return False
        inliers, (i1, i2) = next(iter(self.order_pair.items()))
        matches = self.match_features(i1, i2)
        P1, P2, _, _ = self.get_camera_pose(i1, i2, matches)
        self.camera_poses[i1] = P1
        self.camera_poses[i2] = P2
        self.done_views.update([i1, i2])
        self.good_views.update([i1, i2])
        points3D, ref2D = self.triangulate_points(i1, i2, P1, P2, matches)
        self.recon3Dpts = points3D
        self.rgb_values = get_rgb(self.mat_color_images[i1], ref2D)
        print("=== base reconstruction complete ===")
        return True

    def add_more_views(self):
        for new_view in range(self.num_images):
            if new_view in self.done_views:
                continue
            best_matches, best_done_view, max_match = [], -1, 0
            for done_view in self.done_views:
                matches = self.match_features(done_view, new_view)
                if len(matches) > max_match:
                    max_match, best_matches, best_done_view = len(matches), matches, done_view
            if max_match < 15:
                self.skip_views.add(new_view)
                continue
            pts3d, pts2d = [], []
            for match in best_matches:
                for pt3d in self.recon3Dpts:
                    if best_done_view in pt3d.idx_image and pt3d.idx_image[best_done_view] == match.queryIdx:
                        pts3d.append(pt3d.pt)
                        pts2d.append(self.points2D[new_view][match.trainIdx])
                        break
            if len(pts3d) < 6:
                self.skip_views.add(new_view)
                continue
            pts3d = np.array(pts3d)
            pts2d = np.array(pts2d)
            success, rvec, tvec, inliers = cv2.solvePnPRansac(pts3d, pts2d, self.intrinsic, self.distortion)
            if not success:
                self.skip_views.add(new_view)
                continue
            R, _ = cv2.Rodrigues(rvec)
            self.camera_poses[new_view] = np.hstack((R, tvec))
            self.done_views.add(new_view)
            self.good_views.add(new_view)
            new_pts, aligned2d = self.triangulate_points(best_done_view, new_view,
                                                         self.camera_poses[best_done_view],
                                                         self.camera_poses[new_view], best_matches)
            self.recon3Dpts.extend(new_pts)
            self.rgb_values.extend(get_rgb(self.mat_color_images[best_done_view], aligned2d))
        return True

    def adjust_current_bundle(self, device):
        print(f"Using {device.upper()} for bundle adjustment")
        if device == "cpu":
            BundleAdjustment.adjust_bundle(self.recon3Dpts, self.camera_poses, self.intrinsic, self.distortion, self.points2D)
        else:
            BundleAdjustmentTorch.adjust_bundle(self.recon3Dpts, self.camera_poses, self.intrinsic, self.distortion, self.points2D, device=device)

    def get_cam_position(self):
        self.cam3Dpts = []
        self.cam3DRGB = []
        for i, pose in enumerate(self.camera_poses):
            R = pose[:, :3]
            t = pose[:, 3].reshape((3, 1))
            cam_pos = -R.T @ t
            self.cam_positions.append(cam_pos)
            self.cam3Dpts.append(cam_pos.flatten())
            self.cam3DRGB.append((0., 0., 255.) if i == self.world_index else (0., 255., 0.))

    def get_dir_path(self):
        return self.dir_path

    def map3d(self):
        self.find_best_pair()
        if not self.base_reconstruction():
            return False
        if not self.add_more_views():
            return False
        return True
