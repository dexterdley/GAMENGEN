import torch
from memory import MemoryModule, Map, PlayerState
from observation import ObservationModule
from dynamics import DynamicsModule

class MultiGenEnv:
    """
    The main Inference Environment Loop from Section 3.4.
    Acts as an interactive simulator mapping (State, Action) -> (Next Obs, Next State).
    """
    def __init__(
        self,
        game_map: Map,
        initial_pose: PlayerState,
        context_frames=4,
        resolution=64,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.context_length = context_frames
        self.resolution = resolution
        
        # 1. Initialize Memory Module (State)
        self.memory = MemoryModule(game_map, initial_pose)
        
        # 2. Initialize Observation Module
        self.observation = ObservationModule(
            context_frames=context_frames,
            resolution=resolution
        ).to(device)
        self.observation.eval()
        
        # 3. Initialize Dynamics Module
        self.dynamics = DynamicsModule(
            geometry_dim=resolution # Use the target spatial resolution since disparity is broadcast to this width
        ).to(device)
        self.dynamics.eval()
        
        # 4. Initialize History window
        # We start with empty/black frames for the initial context
        self.visual_context = torch.zeros(
            1, 3 * context_frames, resolution, resolution, device=device
        )
        
    def reset(self):
        """Resets the visual context to 0, returns initial geometric representation."""
        self.visual_context.zero_()
        return self._get_observation_dict()

    def _get_observation_dict(self):
        """Helper to return current observable state for debugging/visualization."""
        geom_rt = self.memory.get_geometric_readout((self.resolution, self.resolution)).to(self.device)
        pose = self.memory.player.get_pose()
        return {
            "pose": pose,
            "geometry_disparity": geom_rt,
            "visual_context": self.visual_context.clone()
        }

    @torch.no_grad()
    def step(self, action_id: int):
        """
        Executes one step of the interactive simulator (Section 3.4).
        (S_t, a_t) |-> (o_{t+1}, S_{t+1})
        """
        # Batch size 1 for inference
        action_tensor = torch.tensor([action_id], device=self.device)
        
        # 1. Query external memory for geometric readout r_t
        geom_rt = self.memory.get_geometric_readout((self.resolution, self.resolution)).to(self.device)
        
        # 2. Sample next frame using Observation model
        # DDPMScheduler handles the reverse process. 
        # For simplicity, we implement a basic fast sampling loop (e.g. DDIM-style or just standard DDPM).
        # We will use the model's standard DDPM scheduler but step through it.
        
        # Start with pure noise
        latents = torch.randn(1, 3, self.resolution, self.resolution, device=self.device)
        
        # We will extract intermediate UNet features at a specific timestep (e.g. half-way)
        # to feed to the Dynamics module as f_t. 
        # For this simulator wrapper, we assume `dynamics` just needs features once.
        f_t = torch.zeros(1, 512, device=self.device) # Fallback / dummy features based on Dynamics in-dim
        
        # Set scheduler for inference
        self.observation.scheduler.set_timesteps(20) # 20 steps for faster inference in simulation
        timesteps = self.observation.scheduler.timesteps
        
        for i, t in enumerate(timesteps):
            t_tensor = torch.tensor([t], device=self.device)
            
            # Predict velocity
            v_pred = self.observation(
                x_noisy=latents,
                timesteps=t_tensor,
                context_frames=self.visual_context,
                geometry_disparity=geom_rt,
                actions=action_tensor
            )
            
            # Convert v-prediction to next noisy sample in the reverse process
            # (Note: In a true implementation, diffusers scheduler.step expects noise prediction or uses step for v_pred)
            latents = self.observation.scheduler.step(v_pred, t, latents).prev_sample
            
            # Extract features for Dynamics early in the denoising process (e.g., step 10)
            if i == len(timesteps) // 2:
                 f_t = self.observation.get_intermediate_features(
                     latents, t_tensor, self.visual_context, geom_rt, action_tensor
                 )

        # The final latent is our predicted frame o_{t+1}
        next_frame = latents.clamp(-1.0, 1.0)
        
        # 3. Update pose using Dynamics Module
        current_pose = torch.tensor([self.memory.player.get_pose()], dtype=torch.float32, device=self.device)
        
        # We need the naked 1D disparity vector for the dynamics model, not the spatial broadcasted one
        # Because we broadcasted it earlier, we can just slice it out or recalculate
        raw_disparity_1d = geom_rt[0, 0, 0, :] # Shape (64,)
        
        delta_pose = self.dynamics(
            pose=current_pose,
            action=action_tensor,
            geometry=raw_disparity_1d.unsqueeze(0),
            unet_features=f_t
        )
        
        updated_pose = self.dynamics.apply_update(current_pose, delta_pose)
        
        # 4. Advance State S_t -> S_{t+1}
        # Update PlayerState
        self.memory.player.x = updated_pose[0, 0].item()
        self.memory.player.y = updated_pose[0, 1].item()
        self.memory.player.yaw = updated_pose[0, 2].item()
        
        # Shift visual context history
        # target frame has 3 channels. Append to the end, drop the front 3.
        self.visual_context = torch.roll(self.visual_context, shifts=-3, dims=1)
        self.visual_context[:, -3:, :, :] = next_frame
        
        return next_frame, self._get_observation_dict()

if __name__ == "__main__":
    print("Testing MultiGenEnv Inference Loop...")
    try:
        test_map = Map.create_simple_box(10.0)
        player = PlayerState(x=5.0, y=5.0, yaw=0.0)
        
        env = MultiGenEnv(test_map, player, context_frames=2, resolution=64, device="cpu")
        env.reset()
        
        print(f"Initial Pose: {env.memory.player.get_pose()}")
        
        # Take a step with dummy action 1
        frame, obs_dict = env.step(action_id=1)
        
        print(f"Generated frame shape: {frame.shape}")
        print(f"New Pose: {obs_dict['pose']}")
        print(f"Loop dry-run successful!")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
