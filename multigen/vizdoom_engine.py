import torch
import numpy as np
import math
import vizdoom as vzd
import os
from memory import PlayerState
from observation import ObservationModule
from dynamics import DynamicsModule

class VizDoomMultiGenEnv:
    """
    An environment wrapper that integrates the real ViZDoom engine (providing true depth and pose)
    with the multigen Observation and Dynamics modules.
    """
    def __init__(
        self,
        scenario_path: str = None,
        context_frames=4,
        resolution=64,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.context_length = context_frames
        self.resolution = resolution
        
        # 1. Initialize ViZDoom Engine
        self.game = vzd.DoomGame()
        if scenario_path is None:
            # Fallback to basic if not provided
            scenario_path = os.path.join(vzd.scenarios_path, "basic.wad")
            
        self.game.set_doom_scenario_path(scenario_path)
        self.game.set_doom_map("map01")
        
        # Low resolution enough for 64x64 or 32x32 scaling
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        
        # Essential buffers and variables
        self.game.set_depth_buffer_enabled(True)
        self.game.set_available_game_variables([
            vzd.GameVariable.POSITION_X,
            vzd.GameVariable.POSITION_Y,
            vzd.GameVariable.ANGLE
        ])
        
        # Available basic actions (MOVE_LEFT, MOVE_RIGHT, ATTACK, JUMP, etc.)
        # Here we just use left, right, forward for simple exploration
        self.available_actions = [
            [True, False, False], # Turn Left
            [False, True, False], # Turn Right
            [False, False, True], # Move Forward
        ]
        self.game.set_available_buttons([
            vzd.Button.TURN_LEFT, 
            vzd.Button.TURN_RIGHT,
            vzd.Button.MOVE_FORWARD
        ])
        
        self.game.set_window_visible(False)
        self.game.init()
        
        # 2. Initialize Observation Module (Multigen diffusers model)
        # Using abstract action_dim = len(available_actions)
        self.observation = ObservationModule(
            context_frames=context_frames,
            action_dim=len(self.available_actions), # e.g. 3
            resolution=resolution
        ).to(device)
        self.observation.eval()
        
        # 3. Initialize Dynamics Module (Predicts pose updates)
        self.dynamics = DynamicsModule(
            geometry_dim=resolution,
            action_dim=len(self.available_actions)
        ).to(device)
        self.dynamics.eval()
        
        # 4. Initialize Context History
        self.visual_context = torch.zeros(
            1, 3 * context_frames, resolution, resolution, device=device
        )
        
        self.current_pose = PlayerState()

    def reset(self):
        self.game.new_episode()
        self.visual_context.zero_()
        return self._get_observation_dict()

    def _extract_geometry(self, state):
        """Extracts the center slice of ViZDoom depth buffer and maps it to disparity spatial tensor."""
        # state.depth_buffer is (H, W) values 0-255 representing depth
        depth = state.depth_buffer
        
        if depth is None:
            # Fallback if depth fails
            return torch.zeros(1, 1, self.resolution, self.resolution, device=self.device)
            
        # Extract the middle horizontal line to simulate a 1D scan
        mid_y = depth.shape[0] // 2
        scanline_1d = depth[mid_y, :] # Shape (160,) if res is 160x120
        
        # Convert depth to disparity
        # Assuming 255 is nearest and 0 is farthest in ViZDoom
        # Or you can do simple normalization. We'll add 1 to avoid div by 0 and invert.
        scanline_1d = np.array(scanline_1d, dtype=np.float32)
        disparity = scanline_1d / 255.0 # Max scaled to 1.0
        
        # Resize to UNet resolution width
        x_old = np.linspace(0, 1, len(disparity))
        x_new = np.linspace(0, 1, self.resolution)
        disparity_resized = np.interp(x_new, x_old, disparity)
        
        # Broadcast to spatial tensor (1, 1, H, W)
        disp_tensor = torch.tensor(disparity_resized, dtype=torch.float32)
        disp_spatial = disp_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 1, self.resolution, 1).to(self.device)
        
        return disp_spatial

    def _extract_pose(self, state):
        """Extracts (x, y, yaw) from ViZDoom."""
        x = state.game_variables[0]
        y = state.game_variables[1]
        # ViZDoom angle is 0-360 degrees. Convert to radians [-pi, pi]
        angle_deg = state.game_variables[2]
        yaw = math.radians(angle_deg)
        yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
        
        self.current_pose.x = x
        self.current_pose.y = y
        self.current_pose.yaw = yaw
        
        return (x, y, yaw)

    def _get_observation_dict(self):
        state = self.game.get_state()
        if state is None:
            return None
            
        geom_rt = self._extract_geometry(state)
        pose = self._extract_pose(state)
        
        return {
            "pose": pose,
            "geometry_disparity": geom_rt,
            "visual_context": self.visual_context.clone()
        }

    @torch.no_grad()
    def step(self, action_id: int):
        """
        Executes one step: steps the engine, extracts state, and runs multigen models.
        """
        # Step ViZDoom environment
        action_booleans = self.available_actions[action_id]
        self.game.make_action(action_booleans, 4) # frameskip 4 to see noticeable visual difference
        
        state = self.game.get_state()
        if state is None:
            # Episode ended
            return None, None
            
        # Get true new state from engine
        geom_rt = self._extract_geometry(state)
        pose = self._extract_pose(state)
        true_pose_tensor = torch.tensor([pose], dtype=torch.float32, device=self.device)
        
        # Action tensor for generator conditioning
        action_tensor = torch.tensor([action_id], device=self.device)
        
        # 1. Sample next frame using Observation model
        latents = torch.randn(1, 3, self.resolution, self.resolution, device=self.device)
        f_t = torch.zeros(1, 512, device=self.device)
        
        self.observation.scheduler.set_timesteps(15) # Faster inference 15 steps
        timesteps = self.observation.scheduler.timesteps
        
        for i, t in enumerate(timesteps):
            t_tensor = torch.tensor([t], device=self.device)
            v_pred = self.observation(
                x_noisy=latents,
                timesteps=t_tensor,
                context_frames=self.visual_context,
                geometry_disparity=geom_rt,
                actions=action_tensor
            )
            latents = self.observation.scheduler.step(v_pred, t, latents).prev_sample
            
            if i == len(timesteps) // 2:
                 f_t = self.observation.get_intermediate_features(
                     latents, t_tensor, self.visual_context, geom_rt, action_tensor
                 )

        next_frame = latents.clamp(-1.0, 1.0)
        
        # 2. Predict Pose using Dynamics Module (to verify it works on real ViZDoom geometry)
        raw_disparity_1d = geom_rt[0, 0, 0, :]
        delta_pose = self.dynamics(
            pose=true_pose_tensor,
            action=action_tensor,
            geometry=raw_disparity_1d.unsqueeze(0),
            unet_features=f_t
        )
        
        predicted_pose = self.dynamics.apply_update(true_pose_tensor, delta_pose)
        
        # 3. Update Visual Context History
        self.visual_context = torch.roll(self.visual_context, shifts=-3, dims=1)
        self.visual_context[:, -3:, :, :] = next_frame
        
        obs_dict = {
            "pose": pose,                  # True pose from ViZDoom
            "predicted_pose": predicted_pose[0].cpu().numpy(), # Predicted by abstract transformer
            "geometry_disparity": geom_rt,
            "visual_context": self.visual_context.clone()
        }
        
        return next_frame, obs_dict

    def close(self):
        self.game.close()
