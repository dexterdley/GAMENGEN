import numpy as np
import torch
import collections
import cv2
from config_multigen import REPLAY_BUFFER_SIZE, CONTEXT_FRAMES, RESOLUTION


class Transition:
    """A single (s_t, a_t) -> s_{t+1} transition with pose supervision."""
    __slots__ = ['target_frame', 'context_frames', 'geometry', 'action',
                 'pose_current', 'pose_next']

    def __init__(self, target_frame, context_frames, geometry, action,
                 pose_current, pose_next):
        self.target_frame = target_frame      # (3, H, W) float32 [-1, 1]
        self.context_frames = context_frames  # (3*L, H, W) float32 [-1, 1]
        self.geometry = geometry              # (1, H, W) float32 disparity
        self.action = action                  # int
        self.pose_current = pose_current      # (3,) float32 [x, y, yaw]
        self.pose_next = pose_next            # (3,) float32 [x, y, yaw]


class ReplayBuffer:
    """
    Circular replay buffer that stores transitions for decoupling
    data collection from gradient updates.
    """
    def __init__(self, capacity=REPLAY_BUFFER_SIZE):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        """Sample a random batch and collate into tensors."""
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        target_frames = torch.stack([t.target_frame for t in batch])
        context_frames = torch.stack([t.context_frames for t in batch])
        geometries = torch.stack([t.geometry for t in batch])
        actions = torch.tensor([t.action for t in batch], dtype=torch.long)
        poses_current = torch.stack([t.pose_current for t in batch])
        poses_next = torch.stack([t.pose_next for t in batch])

        return {
            'target_frame': target_frames,       # (B, 3, H, W)
            'context_frames': context_frames,     # (B, 3*L, H, W)
            'geometry': geometries,               # (B, 1, H, W)
            'action': actions,                    # (B,)
            'pose_current': poses_current,        # (B, 3)
            'pose_next': poses_next,              # (B, 3)
        }

    def __len__(self):
        return len(self.buffer)


def extract_frame_for_diffusion(rgb_buffer, resolution=RESOLUTION):
    """
    Converts ViZDoom RGB uint8 buffer (H, W, 3) to diffusion tensor (3, H, W) in [-1, 1].
    Resizes to target resolution if needed.
    """
    # Resize if needed (e.g. 320x240 → 120x160)
    h, w = rgb_buffer.shape[:2]
    target_h, target_w = resolution
    if h != target_h or w != target_w:
        rgb_buffer = cv2.resize(rgb_buffer, (target_w, target_h), interpolation=cv2.INTER_AREA)

    img_float = rgb_buffer.astype(np.float32) / 255.0
    img_normalized = (img_float * 2.0) - 1.0
    return torch.tensor(img_normalized, dtype=torch.float32).permute(2, 0, 1)


def extract_geometry(depth_buffer, resolution):
    """
    Extracts center scanline from ViZDoom depth buffer → disparity → spatial tensor.

    Args:
        depth_buffer: ViZDoom depth (H, W) uint8
        resolution: tuple (H, W) for output spatial size

    Returns:
        Tensor (1, H, W) disparity
    """
    if depth_buffer is None:
        return torch.zeros(1, resolution[0], resolution[1])

    mid_y = depth_buffer.shape[0] // 2
    scanline_1d = np.array(depth_buffer[mid_y, :], dtype=np.float32)
    disparity = scanline_1d / 255.0  # Normalize

    # Resize to target width
    target_w = resolution[1]
    target_h = resolution[0]
    x_old = np.linspace(0, 1, len(disparity))
    x_new = np.linspace(0, 1, target_w)
    disparity_resized = np.interp(x_new, x_old, disparity)

    # Broadcast to spatial (1, H, W)
    disp_tensor = torch.tensor(disparity_resized, dtype=torch.float32)
    return disp_tensor.unsqueeze(0).repeat(target_h, 1).unsqueeze(0)


def extract_pose(state):
    """
    Extracts (x, y, yaw) from ViZDoom game state.
    Requires POSITION_X, POSITION_Y, ANGLE as game variables.

    Returns:
        Tensor (3,) float32
    """
    import math
    x = state.game_variables[0]
    y = state.game_variables[1]
    angle_deg = state.game_variables[2]
    yaw = math.radians(angle_deg)
    yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
    return torch.tensor([x, y, yaw], dtype=torch.float32)


class ContextTracker:
    """
    Maintains a rolling window of L context frames as a single tensor.
    """
    def __init__(self, context_frames=CONTEXT_FRAMES, resolution=(120, 160)):
        self.L = context_frames
        self.resolution = resolution
        self.buffer = torch.zeros(3 * context_frames, resolution[0], resolution[1])

    def reset(self):
        self.buffer.zero_()

    def push(self, frame):
        """
        Push a new frame (3, H, W) into the context, dropping the oldest.
        """
        self.buffer = torch.roll(self.buffer, shifts=-3, dims=0)
        self.buffer[-3:, :, :] = frame.clone().detach()

    def get(self):
        """Returns current context (3*L, H, W)."""
        return self.buffer.clone()
