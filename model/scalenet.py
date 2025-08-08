# models.py (or scalenet.py)

import gymnasium as gym
import torch
import torch.nn as nn
import timm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ScaleNet(BaseFeaturesExtractor):
    """
    Custom feature extractor using a Vision Transformer (ViT) for image and depth data.
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)

        old_conv = self.vit.patch_embed.proj
        new_conv = nn.Conv2d(4, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride, padding=old_conv.padding)
        
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight.clone()
            new_conv.weight[:, 3, :, :].zero_()
        
        self.vit.patch_embed.proj = new_conv

        vit_output_dim = self.vit.num_features
        self.mlp = nn.Sequential(
            nn.Linear(vit_output_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        # --- THE FIX IS HERE ---
        # VecTransposeImage has already processed the images to (N, C, H, W).
        # We just need to scale them.
        img0 = observations['image0'].float() / 255.0
        img1 = observations['image1'].float() / 255.0
        
        # The depth maps are still (N, H, W). We need to add the channel dimension
        # so they become (N, 1, H, W).
        depth0 = observations['depth0'].unsqueeze(1)
        depth1 = observations['depth1'].unsqueeze(1)

        # Now all tensors have 4 dimensions (N, C, H, W) and can be concatenated.
        rgbd0 = torch.cat([img0, depth0], dim=1) # Concatenates (N,3,H,W) and (N,1,H,W) -> (N,4,H,W)
        rgbd1 = torch.cat([img1, depth1], dim=1)

        features0 = self.vit(rgbd0)
        features1 = self.vit(rgbd1)
        
        combined_features = torch.cat([features0, features1], dim=1)
        
        return self.mlp(combined_features)