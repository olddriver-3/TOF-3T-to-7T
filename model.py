import torch
import torch.nn as nn
from diffusers import UNet2DModel

class ConditionalDiffusionModel(nn.Module):
    def __init__(self, condition_channels=1, output_channels=1, 
                 base_channels=32, image_size=28):
        super().__init__()
        
        self.condition_channels = condition_channels
        self.output_channels = output_channels
        
        self.network = UNet2DModel(
            sample_size=image_size,
            in_channels=output_channels + condition_channels,
            out_channels=output_channels,
            layers_per_block=2,
            block_out_channels=(base_channels, base_channels*2, base_channels*2),
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D", 
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
        )
    
    def forward(self, noisy_data, timestep, condition_data):
        combined_input = torch.cat([noisy_data, condition_data], dim=1)
        return self.network(combined_input, timestep).sample
