import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class BundleAdjustmentTorch:
    def __init__(self, intrinsic, device="mps"):
        self.K = torch.tensor(intrinsic, dtype=torch.float32, device=device)
        self.device = device

    def rodrigues(self, rvec):
        # Batch-compatible Rodrigues' formula in PyTorch
        theta = torch.norm(rvec, dim=1, keepdim=True)
        r = rvec / theta
        r = r.view(-1, 3, 1)
        rt = r.transpose(1, 2)

        cos = torch.cos(theta).view(-1, 1, 1)
        sin = torch.sin(theta).view(-1, 1, 1)
        I = torch.eye(3, device=rvec.device).expand(rvec.shape[0], -1, -1)

        R = cos * I + (1 - cos) * (r @ rt) + sin * self.skew_symmetric(r.squeeze(-1))
        return R

    def skew_symmetric(self, v):
        zero = torch.zeros_like(v[:, 0])
        return torch.stack([
            zero, -v[:, 2], v[:, 1],
            v[:, 2], zero, -v[:, 0],
            -v[:, 1], v[:, 0], zero
        ], dim=1).reshape(-1, 3, 3)

    def project(self, points_3d, rvecs, tvecs):
        R = self.rodrigues(rvecs)
        P = torch.bmm(R, points_3d.transpose(1, 2)) + tvecs.view(-1, 3, 1)
        P = P / P[:, 2:3, :]
        proj_2d = torch.bmm(self.K[None, :2, :], P).squeeze(2)
        return proj_2d

    def run(self, camera_params_init, points_3d_init, camera_indices, point_indices, points_2d_obs,
            n_iters=100, lr=1e-3):

        camera_params = torch.tensor(camera_params_init, dtype=torch.float32, requires_grad=True, device=self.device)
        points_3d = torch.tensor(points_3d_init, dtype=torch.float32, requires_grad=True, device=self.device)

        camera_indices = torch.tensor(camera_indices, dtype=torch.long, device=self.device)
        point_indices = torch.tensor(point_indices, dtype=torch.long, device=self.device)
        points_2d_obs = torch.tensor(points_2d_obs, dtype=torch.float32, device=self.device)

        optimizer = optim.Adam([camera_params, points_3d], lr=lr)

        for iter in range(n_iters):
            optimizer.zero_grad()

            rvecs = camera_params[:, :3]
            tvecs = camera_params[:, 3:].unsqueeze(1)

            pts3d = points_3d[point_indices].unsqueeze(1)
            r = rvecs[camera_indices]
            t = tvecs[camera_indices]
            proj = self.project(pts3d, r, t)

            loss = nn.functional.mse_loss(proj, points_2d_obs)
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print(f"Iter {iter:3d}, Loss: {loss.item():.6f}")

        return camera_params.detach().cpu().numpy(), points_3d.detach().cpu().numpy()
    
    @staticmethod
    def adjust_bundle(point_cloud, camera_poses, intrinsic, distortion, image_2d_features, device="mps"):
        ba = BundleAdjustmentTorch(intrinsic, device=device)

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

        # Run PyTorch-based bundle adjustment
        camera_params_opt, points_3d_opt = ba.run(camera_params, points_3d, camera_indices, point_indices, points_2d)

        # Update camera poses
        for i in range(len(camera_poses)):
            rvec = camera_params_opt[i, :3]
            tvec = camera_params_opt[i, 3:6].reshape(3, 1)
            R, _ = cv2.Rodrigues(rvec)
            camera_poses[i] = np.hstack([R, tvec])

        # Update 3D points
        for i in range(len(point_cloud)):
            point_cloud[i].pt = points_3d_opt[i]

        print("[MPS] Bundle adjustment complete.")    