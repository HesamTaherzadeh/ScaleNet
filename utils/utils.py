# 2_utils.py

import cv2
import numpy as np

orb = cv2.ORB_create(nfeatures=2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def extract_keypoint_matches(img1: np.ndarray, img2: np.ndarray):
    """Extracts ORB keypoints and returns matched points."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return np.array([]), np.array([])

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    
    return pts1, pts2

def get_principal_point(img: np.ndarray):
    """Calculates the principal point from image dimensions."""
    h, w = img.shape[:2]
    return np.array([(w - 1) / 2.0, (h - 1) / 2.0], dtype=np.float32)
    
def get_depths_at_keypoints(depth_map, keypoints, original_dims):
    """Samples depth values at keypoint locations."""
    if keypoints.shape[0] == 0:
        return np.array([])

    h_orig, w_orig = original_dims
    h_depth, w_depth = depth_map.shape

    # Scale keypoints to depth map dimensions
    x_coords = keypoints[:, 0] * (w_depth / w_orig)
    y_coords = keypoints[:, 1] * (h_depth / h_orig)

    # Use integer coordinates for indexing
    x_coords = np.clip(x_coords.astype(int), 0, w_depth - 1)
    y_coords = np.clip(y_coords.astype(int), 0, h_depth - 1)

    return depth_map[y_coords, x_coords]

def compute_rpe(T_gt, T_est):
    """Computes the Relative Pose Error."""
    # Translation error
    trans_gt = T_gt[:3, 3]
    trans_est = T_est[:3, 3]
    trans_error = np.linalg.norm(trans_gt - trans_est)

    # Rotation error
    R_gt = T_gt[:3, :3]
    R_est = T_est[:3, :3]
    R_rel = R_gt.T @ R_est
    trace = np.trace(R_rel)
    # Clip trace to avoid arccos domain errors
    rot_error_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    rot_error_deg = np.degrees(rot_error_rad)
    
    # Combine errors (you can weigh them differently if needed)
    total_error = trans_error + rot_error_deg * 0.1 # Example weighting
    return total_error