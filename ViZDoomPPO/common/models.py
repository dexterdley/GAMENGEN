import gym
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128, **kwargs):
        super().__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.LayerNorm([3, 100, 156]),

            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0, bias=False),
            nn.LayerNorm([32, 24, 38]),
            nn.LeakyReLU(**kwargs),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0, bias=False),
            nn.LayerNorm([64, 11, 18]),
            nn.LeakyReLU(**kwargs),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LayerNorm([64, 9, 16]),
            nn.LeakyReLU(**kwargs),

            nn.Flatten(),
        )

        self.linear = nn.Sequential(
            nn.Linear(9216, features_dim, bias=False),
            nn.LayerNorm(features_dim),
            nn.LeakyReLU(**kwargs),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def init_net(m: nn.Module):
    if len(m._modules) > 0:
        for subm in m._modules:
            init_net(m._modules[subm])
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(
            m.weight,
            a=0.1,  # Same as the leakiness parameter for LeakyReLu.
            mode='fan_in',  # Preserves magnitude in the forward pass.
            nonlinearity='leaky_relu')


class ChannelAttention(nn.Module):
    """
    A lightweight Squeeze-and-Excitation block to act as a visual attention mechanism.
    It helps the model focus on important feature channels (e.g., enemy textures, projectiles).
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        b, c, _, _ = x.size()
        # Calculate attention weights and broadcast back to the feature map
        weights = self.attention(x).view(b, c, 1, 1)
        return x * weights.expand_as(x)

class RobustDoomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256, leaky_slope: float = 0.1):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            # Replaced spatial-dependent LayerNorm with robust GroupNorm
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0, bias=False),
            nn.GroupNorm(8, 32), 
            nn.LeakyReLU(leaky_slope),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0, bias=False),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(leaky_slope),
            
            # Inject visual attention hook here to filter features before the final spatial abstraction
            ChannelAttention(channels=64),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(leaky_slope),
            
            nn.Flatten(),
        )

        # Dynamically compute the flattened shape by doing a dummy forward pass.
        # This prevents manual calculation errors and handles any input resolution.
        with th.no_grad():
            dummy_tensor = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(dummy_tensor).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim, bias=False),
            # LayerNorm is perfectly safe here on a 1D linear tensor
            nn.LayerNorm(features_dim),
            nn.LeakyReLU(leaky_slope)
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def init_model(model):
    init_net(model.policy)
