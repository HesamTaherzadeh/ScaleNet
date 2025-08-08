# train.py

import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder, VecTransposeImage
from stable_baselines3.common.callbacks import ProgressBarCallback

from env.rl_env import ScaleEnv
from model.scalenet import ScaleNet

def main():
    parser = argparse.ArgumentParser(description="Train PPO with pre-processed data.")
    # --- Arguments are now simpler ---
    parser.add_argument("--data_file", type=str, default="preprocessed_data.pkl", help="Path to the pre-processed data file.")
    parser.add_argument("--log_dir", type=str, default="ppo_kitti_logs", help="Directory to save logs and models")
    parser.add_argument("--video_dir", type=str, default="ppo_videos", help="Directory to save videos of episodes")
    parser.add_argument("--total_steps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--n_envs", type=int, default=25, help="Number of parallel environments")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)

    # --- env_kwargs now points to the pre-processed file ---
    env_kwargs = {
        'preprocessed_data_path': args.data_file,
        'episode_length': 15,
        'render_mode': 'rgb_array'
    }

    def make_env():
        return ScaleEnv(**env_kwargs)

    vec_env = SubprocVecEnv([make_env for _ in range(args.n_envs)])
    vec_env = VecVideoRecorder(vec_env, args.video_dir,
                               record_video_trigger=lambda x: x % 5000 == 0, name_prefix="ppo-run")
    vec_env = VecTransposeImage(vec_env)
                                
    policy_kwargs = dict(
        features_extractor_class=ScaleNet,
        features_extractor_kwargs=dict(features_dim=256),
    )

    model = PPO( "MultiInputPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=0,
                 tensorboard_log=args.log_dir, learning_rate=1e-5, n_steps=2048,
                 batch_size=64, gamma=0.98, gae_lambda=0.95, clip_range=0.2 )

    print(f"--- Starting PPO Training with pre-processed data and {args.n_envs} parallel environments ---")
    model.learn( total_timesteps=args.total_steps, callback=ProgressBarCallback() )
    print("\n--- Training Finished ---")

    model_save_path = os.path.join(args.log_dir, "ppo_scalenet_final.zip")
    model.save(model_save_path)
    
    vec_env.close()

if __name__ == '__main__':
    main()