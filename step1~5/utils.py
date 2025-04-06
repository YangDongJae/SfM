# myutils.py - Utility functions for SfM pipeline

import os
import numpy as np
import cv2
from glob import glob
from sklearn.cluster import DBSCAN

def read_directory(path):
    valid_exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG",".HEIC")
    return sorted([f for f in os.listdir(path) if f.endswith(valid_exts)])

def multiply_matrix(mat1, mat2):
    return np.dot(mat1, mat2)

def plus_matrix(mat1, mat2):
    return np.add(mat1, mat2)

def diag_matrix(mat):
    return np.diag(np.diag(mat))

def multiply_element(mat1, mat2):
    return np.multiply(mat1, mat2)

def double_matrix_element(mat, row=None, col=None):
    if row is not None:
        return np.square(mat[row:row+1, :])
    elif col is not None:
        return np.square(mat[:, col:col+1])
    else:
        raise ValueError("Row or column index must be provided")

def divide_matrix_element(mat1, mat2):
    return np.divide(mat1, mat2)

def split_matrix(mat, start_row):
    return mat[start_row:start_row+3, :]

def epipolar_line_image(result_image, image1, image2, points1, points2, fundamental_matrix, which_image):
    lines = cv2.computeCorrespondEpilines(np.array(points1 if which_image == 1 else points2).reshape(-1,1,2), which_image, fundamental_matrix)
    lines = lines.reshape(-1, 3)
    for r in lines:
        x0, y0 = 0, int(-r[2]/r[1])
        x1, y1 = image2.shape[1], int(-(r[2] + r[0]*image2.shape[1])/r[1])
        cv2.line(result_image, (x0, y0), (x1, y1), (255, 255, 255))

def get_rgb(image, points):
    rgb_list = []
    for pt in points:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            bgr = image[y, x]
            rgb_list.append((float(bgr[2]), float(bgr[1]), float(bgr[0])))
    return rgb_list

def save_point_cloud_ply(filename, path, obj_point, obj_rgb, cam_point, cam_rgb):
    full_path = os.path.join(path, filename)
    with open(full_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(obj_point) + len(cam_point)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar diffuse_blue\n")
        f.write("property uchar diffuse_green\n")
        f.write("property uchar diffuse_red\n")
        f.write("end_header\n")

        if len(obj_rgb) == 0:
            obj_rgb = [(255, 255, 255)] * len(obj_point)

        for pt, color in zip(obj_point, obj_rgb):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {int(color[2])} {int(color[1])} {int(color[0])}\n")

        for pt, color in zip(cam_point, cam_rgb):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {int(color[2])} {int(color[1])} {int(color[0])}\n")

    print(f"Saved PLY file: {full_path} (obj point: {len(obj_point)}, camera point: {len(cam_point)})")
    
def remove_outliers_dbscan(points3d, eps=0.2, min_samples=10):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points3d)
    labels = clustering.labels_

    inlier_mask = labels != -1
    cleaned_points = points3d[inlier_mask]
    
    print(f"Before: {len(points3d)} point, After DBSCAN: {len(cleaned_points)} point")
    return cleaned_points, inlier_mask