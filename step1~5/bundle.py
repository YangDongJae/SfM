# bundle.py - Bundle Adjustment using scipy.optimize (as replacement for Ceres)

import numpy as np
from scipy.optimize import least_squares
import cv2

class Point3D:
    def __init__(self, pt):
        self.pt = pt  # np.array([x, y, z])
        self.idx_image = {}  # camera index -> 2D feature index
        self.pt2D = {}  # camera index -> 2D point

class BundleAdjustment:
    def __init__(self):
        pass

    @staticmethod
    def rodrigues_rotation_matrix(rvec):
        R, _ = cv2.Rodrigues(rvec)
        return R

    @staticmethod
    def project(points_3d, rvec, tvec, K):
        R = BundleAdjustment.rodrigues_rotation_matrix(rvec)
        proj_points = (R @ points_3d.T + tvec.reshape(3,1)).T
        proj_points /= proj_points[:, 2].reshape(-1, 1)
        proj_points_2d = (K @ proj_points.T).T[:, :2]
        return proj_points_2d

    @staticmethod
    def reprojection_residuals(params, n_cams, n_points, camera_indices, point_indices, points_2d, K):
        camera_params = params[:n_cams * 6].reshape((n_cams, 6))
        points_3d = params[n_cams * 6:].reshape((n_points, 3))
        residuals = []
        for i in range(len(points_2d)):
            cam_idx = camera_indices[i]
            pt_idx = point_indices[i]
            rvec = camera_params[cam_idx, :3]
            tvec = camera_params[cam_idx, 3:6]
            pt3d = points_3d[pt_idx].reshape(1, 3)
            proj = BundleAdjustment.project(pt3d, rvec, tvec, K)
            error = (proj[0] - points_2d[i])
            residuals.append(error)
        return np.array(residuals).ravel()

    @staticmethod
    def adjust_bundle(point_cloud, camera_poses, intrinsic, distortion, image_2d_features):
        # Initial camera parameters (angle-axis + translation)
        camera_params = []
        for pose in camera_poses:
            if pose is None:
                camera_params.append(np.zeros(6))
                continue
            R = pose[:3, :3]
            t = pose[:3, 3]
            rvec, _ = cv2.Rodrigues(R)
            camera_params.append(np.hstack([rvec.ravel(), t]))

        camera_params = np.array(camera_params)

        # Initial 3D points
        points_3d = np.array([pt.pt for pt in point_cloud])

        # Create observations
        camera_indices = []
        point_indices = []
        points_2d = []
        for i, pt in enumerate(point_cloud):
            for cam_idx, feat_idx in pt.idx_image.items():
                pt2d = image_2d_features[cam_idx][feat_idx]
                points_2d.append(np.array(pt2d))
                camera_indices.append(cam_idx)
                point_indices.append(i)

        # Bundle optimization
        x0 = np.hstack([camera_params.ravel(), points_3d.ravel()])
        res = least_squares(BundleAdjustment.reprojection_residuals, x0,
                            verbose=2,
                            x_scale='jac',
                            ftol=1e-4, method='trf',
                            args=(len(camera_poses), len(point_cloud), camera_indices, point_indices, points_2d, intrinsic))

        # Update camera poses
        camera_params_opt = res.x[:len(camera_poses) * 6].reshape((len(camera_poses), 6))
        for i in range(len(camera_poses)):
            rvec = camera_params_opt[i, :3]
            tvec = camera_params_opt[i, 3:6].reshape(3, 1)
            R = BundleAdjustment.rodrigues_rotation_matrix(rvec)
            camera_poses[i] = np.hstack([R, tvec])

        # Update 3D points
        points_3d_opt = res.x[len(camera_poses) * 6:].reshape((len(point_cloud), 3))
        for i in range(len(point_cloud)):
            point_cloud[i].pt = points_3d_opt[i]

        print("Bundle adjustment complete.")
