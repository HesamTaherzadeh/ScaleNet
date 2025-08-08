# preprocess_data.py

import os
import argparse
import pickle
import cv2
import numpy as np
from tqdm import tqdm

from utils.kitti_helper import KITTIHelperDataset
from mde_models.unidepth import UniDepthONNX
from utils.utils import extract_keypoint_matches

def normalize_depth(d_map):
    d_min = d_map.min()
    d_max = d_map.max()
    if d_max == d_min: return np.zeros_like(d_map)
    return (d_map - d_min) / (d_max - d_min)

def main():
    parser = argparse.ArgumentParser(description="Pre-process KITTI data for faster RL training.")
    parser.add_argument("--kitti_dir", type=str, required=True)
    parser.add_argument("--sequences", nargs='+', default=['05'])
    parser.add_argument("--depth_model", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="preprocessed_ram.pkl")
    args = parser.parse_args()

    dataset = KITTIHelperDataset(args.kitti_dir, args.sequences, args.depth_model)
    depth_model = UniDepthONNX(args.depth_model, image_width=644, image_height=364)
    
    print(f"Found {len(dataset)} total frames. Pre-processing pairs...")
    
    all_pairs = []
    obs_h, obs_w = 224, 224

    for i in tqdm(range(len(dataset) - 1)):
        img0, pose0_gt = dataset[i]
        img1, pose1_gt = dataset[i+1]
        
        path0_parts = dataset.image_paths[i].split(os.sep)
        path1_parts = dataset.image_paths[i+1].split(os.sep)
        if path0_parts[-3] != path1_parts[-3]: continue

        depth0_full = depth_model(img0)
        depth1_full = depth_model(img1)
        mkpts0, mkpts1 = extract_keypoint_matches(img0, img1)
        
        if mkpts0.shape[0] < 8: continue
            
        # --- Store final, processed numpy arrays directly ---
        pair_data = {
            # Data for the observation dictionary (already resized and normalized)
            "obs_img0": cv2.resize(img0, (obs_w, obs_h)),
            "obs_depth0": normalize_depth(cv2.resize(depth0_full, (obs_w, obs_h))).astype(np.float16),
            
            # Full resolution data needed for madpose and rendering
            "render_img": img0,
            "depth0": depth0_full.astype(np.float16),
            "depth1": depth1_full.astype(np.float16),
            "mkpts0": mkpts0.astype(np.float16),
            "mkpts1": mkpts1.astype(np.float16),
            "pose0_gt": pose0_gt,
            "pose1_gt": pose1_gt,
        }
        all_pairs.append(pair_data)
        
    print(f"Saving {len(all_pairs)} valid pairs to {args.output_file}...")
    with open(args.output_file, 'wb') as f:
        pickle.dump(all_pairs, f)
    print("Done.")

if __name__ == '__main__':
    main()