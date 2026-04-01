import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from diffusers import DDPMScheduler

try:
    from config_multigen import CONTEXT_NOISE_SCALE
except ImportError:
    CONTEXT_NOISE_SCALE = 0.1

class ObservationModule(nn.Module):
    """
    Observation Module from Section 3.2 combining pre-trained UNet concepts
    with custom conditioning for Game Engine states.
    """
    def __init__(
        self,
        in_channels_frame=3,    # RGB
        context_frames=4,       # L context frames
        in_channels_geometry=1, # 1D disparity broadcasted to spatial
        action_dim=16,          # Dimensions corresponding to the number of discrete actions
        resolution=(120, 160),  # Spatial resolution for the generator (H, W)
    ):
        super().__init__()
        self.context_frames = context_frames
        self.resolution = resolution
        
        # The input to the UNet at time t is:
        # [Noised Target Frame (3)] + [L Context Frames (3 * L)] + [Disparity Geometry (1)]
        unet_in_channels = in_channels_frame + (context_frames * in_channels_frame) + in_channels_geometry
        
        # We use a standard UNet2DConditionModel from diffusers.
        # We specify cross_attention_dim = action_dim to allow the model to be conditioned on the discrete action embedding
        
        # Ensure resolution is a tuple of (H, W)
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
            
        self.unet = UNet2DConditionModel(
            sample_size=resolution,
            in_channels=unet_in_channels,
            out_channels=in_channels_frame, # Predicts noise/velocity for the target frame
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            cross_attention_dim=action_dim, 
        )
        
        # Action embedding: maps discrete actions (e.g., 0-15) to continuous vectors
        self.action_embedding = nn.Embedding(num_embeddings=action_dim, embedding_dim=action_dim)
        
        # Standard noise scheduler for training and inference
        # The paper mentions 'velocity-parameterization objective' (v-prediction), so we configure the scheduler for it.
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            prediction_type="v_prediction"
        )
        
    def _prepare_inputs(self, x_noisy, context_frames, geometry_disparity):
        """
        Concatenates the noised frame, historical context, and geometric map.
        
        Args:
            x_noisy: (B, 3, H, W)
            context_frames: (B, 3*L, H, W)
            geometry_disparity: (B, 1, H, W)
        Returns:
            unet_input: (B, 3 + 3*L + 1, H, W)
        """
        return torch.cat([x_noisy, context_frames, geometry_disparity], dim=1)

    def forward(self, x_noisy, timesteps, context_frames, geometry_disparity, actions):
        """
        Forward pass for the UNet.
        
        Args:
            x_noisy: Tensor (B, 3, H, W) representing the noised target frame o_{t+1}
            timesteps: Tensor (B,) diffusion timesteps
            context_frames: Tensor (B, 3*L, H, W) the past L frames (optionally noised during training)
            geometry_disparity: Tensor (B, 1, H, W) the ray-traced disparity map
            actions: Tensor (B,) discrete action indices
            
        Returns:
            predicted_velocity: Tensor (B, 3, H, W)
        """
        # Embed the action
        # The shape expected by diffusers cross-attention is (B, sequence_length, cross_attention_dim)
        # We treat the action as a sequence of length 1.
        act_emb = self.action_embedding(actions).unsqueeze(1) # (B, 1, action_dim)
        
        # Prepare concatenated input tensor
        unet_input = self.prepare_unet_inputs(x_noisy, context_frames, geometry_disparity)
        
        # Query UNet
        # We also want to extract intermediate features for the Dynamics module
        # Using `return_dict=False` allows us to just grab the output tensor
        pred = self.unet(
            sample=unet_input,
            timestep=timesteps,
            encoder_hidden_states=act_emb,
            return_dict=False
        )[0]
        
        return pred
        
    def prepare_unet_inputs(self, x_noisy, context_frames, geometry_disparity):
        return torch.cat([x_noisy, context_frames, geometry_disparity], dim=1)

    def compute_loss(self, target_frame, context_frames, geometry_disparity, actions, context_noise_scale=0.1):
        """
        Computes the training loss with "Noised-Context Training for drift robustness" (Section 3.2).
        """
        batch_size = target_frame.shape[0]
        device = target_frame.device
        
        # 1. Sample diffusion timesteps
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), dtype=torch.long, device=device)
        
        # 2. Add noise to the target frame according to the forward diffusion process
        noise = torch.randn_like(target_frame)
        x_noisy = self.scheduler.add_noise(target_frame, noise, timesteps)
        
        # 3. Velocity parameterization target (v-prediction)
        # diffusers DDPMScheduler provides the 'get_velocity' helper to compute the v target
        v_target = self.scheduler.get_velocity(target_frame, noise, timesteps)
        
        # 4. Noised-Context Augmentation
        # Add a random amount of noise to the *context* frames to force the network to rely on the geometry
        # and ignore slight corruptions in autoregressive histories.
        if self.training and context_noise_scale > 0.0:
            # Sample a single noise level in [0, context_noise_scale] for the entire batch
            noise_level = torch.rand(1, device=device) * context_noise_scale
            ctx_noise = torch.randn_like(context_frames) * noise_level.view(-1, 1, 1, 1)
            context_frames_noisy = context_frames + ctx_noise
            # Clamp to valid image ranges assuming [-1, 1] normalization
            context_frames_noisy = torch.clamp(context_frames_noisy, -1.0, 1.0)
        else:
            context_frames_noisy = context_frames
            
        # 5. Predict velocity
        v_pred = self.forward(x_noisy, timesteps, context_frames_noisy, geometry_disparity, actions)
        
        # 6. L2 Loss
        loss = nn.functional.mse_loss(v_pred, v_target)
        
        return loss

    def compute_loss_with_noised_context(
        self, target_frame, context_frames, geometry_disparity, actions,
        max_noise_scale=None
    ):
        """
        Training loss with Noised-Context Augmentation (Section 3.2.4).
        Adds random Gaussian noise to context frames to reduce train-test mismatch
        in autoregressive rollouts.

        Returns:
            loss: scalar diffusion loss
            v_pred: predicted velocity (for logging)
        """
        if max_noise_scale is None:
            max_noise_scale = CONTEXT_NOISE_SCALE

        batch_size = target_frame.shape[0]
        device = target_frame.device

        # 1. Sample diffusion timesteps
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (batch_size,), dtype=torch.long, device=device
        )

        # 2. Noise the target frame
        noise = torch.randn_like(target_frame)
        x_noisy = self.scheduler.add_noise(target_frame, noise, timesteps)

        # 3. V-prediction target
        v_target = self.scheduler.get_velocity(target_frame, noise, timesteps)

        # 4. Noised-Context Augmentation (Section 3.2.4)
        if self.training and max_noise_scale > 0.0:
            noise_level = torch.rand(1, device=device) * max_noise_scale
            ctx_noise = torch.randn_like(context_frames) * noise_level.view(-1, 1, 1, 1)
            context_frames_noisy = torch.clamp(context_frames + ctx_noise, -1.0, 1.0)
        else:
            context_frames_noisy = context_frames

        # 5. Predict velocity
        v_pred = self.forward(
            x_noisy, timesteps, context_frames_noisy,
            geometry_disparity, actions
        )

        # 6. L2 Loss
        loss = nn.functional.mse_loss(v_pred, v_target)

        return loss, v_pred

    def get_intermediate_features(self, x_noisy, timesteps, context_frames, geometry_disparity, actions):
         """
         Helper to run a forward pass and extract the mid-block features to feed to the Dynamics module.
         """
         act_emb = self.action_embedding(actions).unsqueeze(1)
         unet_input = self.prepare_unet_inputs(x_noisy, context_frames, geometry_disparity)
         
         # Note: extracting intermediate features from diffusers UNet generically is tricky.
         # For simplicity in this implementation, we will pass the inputs through downblocks -> midblock,
         # return the midblock representation, and skip the upblocks.
         
         b_size = unet_input.shape[0]
         
         # time embeddings
         t_emb = self.unet.time_proj(timesteps)
         t_emb = t_emb.to(dtype=self.unet.dtype)
         emb = self.unet.time_embedding(t_emb)
         
         # Initial conv
         sample = self.unet.conv_in(unet_input)
         
         # Down blocks
         down_block_res_samples = (sample,)
         for downsample_block in self.unet.down_blocks:
             if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                 sample, res_samples = downsample_block(
                     hidden_states=sample, temb=emb, encoder_hidden_states=act_emb
                 )
             else:
                 sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
             down_block_res_samples += res_samples
             
         # Mid block
         sample = self.unet.mid_block(
             sample, temb=emb, encoder_hidden_states=act_emb
         )
         
         # We'll pool the midblock representation temporally/spatially to create feature vector `f_t`
         pooled_features = nn.functional.adaptive_avg_pool2d(sample, (1, 1)).view(b_size, -1)
         # Detach so dynamics gradients don't flow back into observation UNet
         return pooled_features.detach()


if __name__ == "__main__":
    # Simple validation script
    print("Testing ObservationModule initialization and forward pass...")
    try:
        B, L, H, W = 2, 4, 120, 160
        model = ObservationModule(resolution=(H, W))
        
        target_frame = torch.randn(B, 3, H, W)
        context_frames = torch.randn(B, 3*L, H, W)
        geometry = torch.randn(B, 1, H, W)
        actions = torch.randint(0, 16, (B,))
        
        loss = model.compute_loss(target_frame, context_frames, geometry, actions)
        print(f"Loss computed successfully: {loss.item():.4f}")
        
        # Test feature extraction:
        timesteps = torch.tensor([500, 500])
        x_noisy = torch.randn(B, 3, H, W)
        features = model.get_intermediate_features(x_noisy, timesteps, context_frames, geometry, actions)
        print(f"Intermediate features shape extracted successfully: {features.shape}")
        
    except Exception as e:
        print(f"Error during validation: {e}")
