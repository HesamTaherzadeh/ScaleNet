# kitti_dataset.py

import os
import numpy as np
import cv2
import pandas as pd
from torch.utils.data import Dataset

class KITTIHelperDataset(Dataset):
    """
    A simple dataset helper that provides an image and its corresponding
    ground truth 4x4 pose matrix for a given index.
    """
    def __init__(self, kitti_base_dir: str, sequence_names: list[str], depth_model_path: str):
        self.kitti_base_dir = kitti_base_dir
        self.sequence_names = sequence_names
        self.image_paths, self.poses = self._load_sequences()
        
    def _load_sequences(self):
        all_image_paths = []
        all_poses = []

        for seq_name in self.sequence_names:
            seq_dir = os.path.join(self.kitti_base_dir, 'sequences', seq_name)
            image_dir = os.path.join(seq_dir, 'image_2')
            pose_file = os.path.join(self.kitti_base_dir, 'poses', f'{seq_name}.txt')
            
            if not os.path.exists(pose_file):
                print(f"WARN: Pose file not found for sequence {seq_name}, skipping.")
                continue

            current_image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
            current_poses = pd.read_csv(pose_file, sep=' ', header=None).values
            
            min_len = min(len(current_poses), len(current_image_paths))
            if len(current_poses) != len(current_image_paths):
                 print(f"WARN: Mismatch in sequence {seq_name}. Truncating to {min_len} frames.")

            all_image_paths.extend(current_image_paths[:min_len])
            all_poses.extend(current_poses[:min_len].tolist())

        return all_image_paths, np.array(all_poses)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        This method must return exactly two values: the image and the pose.
        """
        # Load the image
        image_path = self.image_paths[idx]
        img = cv2.imread(image_path)
        
        # Get the 3x4 pose and convert to a 4x4 homogeneous matrix
        pose_3x4 = self.poses[idx].reshape(3, 4)
        pose_4x4 = np.eye(4, dtype=np.float32)
        pose_4x4[:3, :] = pose_3x4
        
        # Return only the image and the 4x4 pose
        return img, pose_4x4