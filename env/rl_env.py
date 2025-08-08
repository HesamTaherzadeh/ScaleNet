# env/rl_env.py

import gymnasium as gym
import numpy as np
import cv2
import madpose
import pickle
from madpose.utils import get_depths

from utils.utils import get_principal_point, compute_rpe

def normalize_depth(d_map):
    d_min = d_map.min()
    d_max = d_map.max()
    if d_max == d_min:
        return np.zeros_like(d_map)
    return (d_map - d_min) / (d_max - d_min)

class ScaleEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, preprocessed_data_path: str, episode_length: int = 15, render_mode: str | None = None):
        super().__init__()
        
        print(f"Loading pre-processed data from {preprocessed_data_path}...")
        with open(preprocessed_data_path, 'rb') as f:
            self.pairs = pickle.load(f)
        print(f"Loaded {len(self.pairs)} data pairs.")

        self.episode_length = episode_length
        self.render_mode = render_mode
        self.action_space = gym.spaces.Box(low=np.array([0.1, -5.0]), high=np.array([20.0, 5.0]), dtype=np.float32)

        obs_h, obs_w = 224, 224
        self.observation_space = gym.spaces.Dict({
            "image0": gym.spaces.Box(low=0, high=255, shape=(obs_h, obs_w, 3), dtype=np.uint8),
            "image1": gym.spaces.Box(low=0, high=255, shape=(obs_h, obs_w, 3), dtype=np.uint8),
            "depth0": gym.spaces.Box(low=0, high=1, shape=(obs_h, obs_w), dtype=np.float32),
            "depth1": gym.spaces.Box(low=0, high=1, shape=(obs_h, obs_w), dtype=np.float32),
        })
        
        self._setup_madpose()
        self.current_pair_idx = 0
        self.steps_in_episode = 0
        self.accumulated_pose_est = np.eye(4)
        self.start_pose_gt = np.eye(4)
        self.current_image_for_render = None
        
    def render(self):
        if self.render_mode == "rgb_array":
            return self.current_image_for_render
        
    def _setup_madpose(self):
            reproj_pix_thres = 8.0
            epipolar_pix_thres = 2.0
            epipolar_weight = 1.0
            self.ransac_opts = madpose.HybridLORansacOptions()
            self.ransac_opts.min_num_iterations = 100
            self.ransac_opts.max_num_iterations = 1000
            self.ransac_opts.final_least_squares = True
            self.ransac_opts.threshold_multiplier = 5.0
            self.ransac_opts.num_lo_steps = 4
            self.ransac_opts.squared_inlier_thresholds = [reproj_pix_thres**2, epipolar_pix_thres**2]
            self.ransac_opts.data_type_weights = [1.0, epipolar_weight]
            self.ransac_opts.random_seed = 0

            self.est_cfg = madpose.EstimatorConfig()
            self.est_cfg.min_depth_constraint = True
            self.est_cfg.use_shift = False
            self.est_cfg.ceres_num_threads = 8

    def _get_obs_and_data(self, idx):
        pair = self.pairs[idx]
        
        img0 = cv2.imread(pair["img0_path"])
        self.current_image_for_render = img0
        
        unnormalized_depth0 = pair["depth0"].astype(np.float32)
        
        obs_h, obs_w = 224, 224
        # For the observation, we also need the *next* frame's data
        next_pair = self.pairs[idx + 1]
        img1 = cv2.imread(next_pair["img0_path"]) # The next pair's first image is this pair's second image
        unnormalized_depth1 = next_pair["depth0"].astype(np.float32)

        obs = {
            "image0": cv2.resize(img0, (obs_w, obs_h)),
            "image1": cv2.resize(img1, (obs_w, obs_h)),
            "depth0": normalize_depth(cv2.resize(unnormalized_depth0, (obs_w, obs_h))),
            "depth1": normalize_depth(cv2.resize(unnormalized_depth1, (obs_w, obs_h)))
        }
        
        data = {
            "img0": img0, "img1": img1,
            "mkpts0": pair["mkpts0"].astype(np.float32), "mkpts1": pair["mkpts1"].astype(np.float32),
            "depth0": unnormalized_depth0, "depth1": pair["depth1"].astype(np.float32),
            "pp0": get_principal_point(img0), "pp1": get_principal_point(img1),
            "T_gt_relative": np.linalg.inv(pair["pose0_gt"]) @ pair["pose1_gt"]
        }
        return obs, data

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        max_start_idx = len(self.pairs) - self.episode_length - 2
        self.current_pair_idx = self.np_random.integers(0, max_start_idx)
        self.steps_in_episode = 0
        self.accumulated_pose_est = np.eye(4)
        self.start_pose_gt = self.pairs[self.current_pair_idx]["pose0_gt"]
        obs, self.step_data = self._get_obs_and_data(self.current_pair_idx)
        return obs, {}

    def step(self, action):
        a, b = action
        data = self.step_data
        metric_depth0 = a * data["depth0"] + b
        metric_depth1 = a * data["depth1"] + b
        d0_kp = get_depths(data["img0"], metric_depth0, data["mkpts0"])
        d1_kp = get_depths(data["img1"], metric_depth1, data["mkpts1"])
        
        if data["mkpts0"].shape[0] < 8:
             T_est_relative = np.eye(4); T_est_relative[0, 3] = 1000 
        else:
            try:
                pose, _ = madpose.HybridEstimatePoseScaleOffsetSharedFocal(
                    data["mkpts0"], data["mkpts1"], d0_kp, d1_kp,
                    [float(metric_depth0.min()), float(metric_depth1.min())],
                    data["pp0"], data["pp1"], self.ransac_opts, self.est_cfg)
                T_est_relative = np.eye(4); T_est_relative[:3, :3] = pose.R(); T_est_relative[:3, 3] = pose.t()
            except Exception:
                T_est_relative = np.eye(4); T_est_relative[0, 3] = 999
        
        self.accumulated_pose_est = self.accumulated_pose_est @ T_est_relative
        self.steps_in_episode += 1
        terminated = self.steps_in_episode >= self.episode_length
        info = {}
        
        if terminated:
            final_pose_gt = self.pairs[self.current_pair_idx]["pose0_gt"]
            total_gt_transform = np.linalg.inv(self.start_pose_gt) @ final_pose_gt
            reward = -1000.0 if np.any(self.accumulated_pose_est > 100) else -compute_rpe(total_gt_transform, self.accumulated_pose_est)
            obs, _ = self._get_obs_and_data(self.current_pair_idx)
            info["final_observation"] = obs
        else:
            self.current_pair_idx += 1
            obs, self.step_data = self._get_obs_and_data(self.current_pair_idx)
            reward = 0.0

        return obs, reward, terminated, False, info