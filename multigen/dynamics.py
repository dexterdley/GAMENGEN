import torch
import torch.nn as nn
import math

try:
    from config_multigen import (
        ACTION_DIM, UNET_MID_FEATURES_DIM, RESOLUTION,
        DYNAMICS_D_MODEL, DYNAMICS_NHEAD, DYNAMICS_NUM_LAYERS,
        DYNAMICS_POSE_DIM,
    )
except ImportError:
    ACTION_DIM = 8
    UNET_MID_FEATURES_DIM = 512
    RESOLUTION = (120, 160)
    DYNAMICS_D_MODEL = 128
    DYNAMICS_NHEAD = 4
    DYNAMICS_NUM_LAYERS = 2
    DYNAMICS_POSE_DIM = 3

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (B, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]

class DynamicsModule(nn.Module):
    """
    Dynamics Module (Section 3.3).
    A lightweight transformer encoder that predicts incremental pose updates.
    """
    def __init__(
        self,
        pose_dim=DYNAMICS_POSE_DIM,
        action_dim=ACTION_DIM,
        geometry_dim=RESOLUTION[1],        # 1D disparity vector width (160)
        unet_feature_dim=UNET_MID_FEATURES_DIM,  # pooled UNet mid-block (512)
        d_model=DYNAMICS_D_MODEL,
        nhead=DYNAMICS_NHEAD,
        num_layers=DYNAMICS_NUM_LAYERS,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Projection layers to map different input modalities to the same d_model
        self.proj_pose = nn.Linear(pose_dim, d_model)
        self.action_embedding = nn.Embedding(action_dim, d_model)
        self.proj_geometry = nn.Linear(geometry_dim, d_model)
        self.proj_unet_features = nn.Linear(unet_feature_dim, d_model)
        
        # Transformer sequence components
        self.pos_encoder = PositionalEncoding(d_model, max_len=4) # 4 input tokens
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head predicts delta position (dx, dy) and delta yaw (dyaw)
        # We pool the transformer outputs and project to 3 dims
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3) 
        )

    def forward(self, pose, action, geometry, unet_features):
        """
        Args:
            pose: Tensor (B, 3) -> [x, y, yaw]
            action: Tensor (B,) -> discrete action indices
            geometry: Tensor (B, num_rays) -> 1D disparity vector
            unet_features: Tensor (B, unet_feature_dim) -> Spatial pooling of UNet bottleneck
            
        Returns:
            delta_pose: Tensor (B, 3) -> [dx, dy, dyaw]
        """
        B = pose.shape[0]
        
        # Project all inputs to d_model
        emb_pose = self.proj_pose(pose).unsqueeze(1)                   # (B, 1, d_model)
        emb_action = self.action_embedding(action).unsqueeze(1)        # (B, 1, d_model)
        emb_geom = self.proj_geometry(geometry).unsqueeze(1)           # (B, 1, d_model)
        emb_unet = self.proj_unet_features(unet_features).unsqueeze(1) # (B, 1, d_model)
        
        # Concatenate into a sequence: sequence length = 4
        # Our "tokens" are: [Pose, Action, Geometry, UNet_Features]
        seq = torch.cat([emb_pose, emb_action, emb_geom, emb_unet], dim=1) # (B, 4, d_model)
        
        # Add positional encoding
        seq = self.pos_encoder(seq)
        
        # Pass through Transformer
        out_seq = self.transformer(seq) # (B, 4, d_model)
        
        # Aggregate sequence (e.g., mean pooling over the 4 tokens)
        pooled = out_seq.mean(dim=1) # (B, d_model)
        
        # Predict delta
        delta_pose = self.head(pooled) # (B, 3) -> (dx, dy, dyaw)
        
        return delta_pose

    @staticmethod
    def wrap_angle(angle):
        """Wraps angle to [-pi, pi]"""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def apply_update(self, current_pose, delta_pose):
        """
        Applies the predicted delta to the current pose.
        Args:
            current_pose: (B, 3) [x, y, yaw]
            delta_pose: (B, 3) [dx, dy, dyaw]
        Returns:
            new_pose: (B, 3)
        """
        new_x = current_pose[:, 0] + delta_pose[:, 0]
        new_y = current_pose[:, 1] + delta_pose[:, 1]
        new_yaw = self.wrap_angle(current_pose[:, 2] + delta_pose[:, 2])
        
        return torch.stack([new_x, new_y, new_yaw], dim=1)

    def compute_loss(self, current_pose, action, geometry, unet_features, next_pose_target):
        """
        Supervises the dynamics module. L2 loss on translation and wrapped angle error on orientation.
        """
        delta_pose_pred = self.forward(current_pose, action, geometry, unet_features)
        
        # Target deltas
        dx_target = next_pose_target[:, 0] - current_pose[:, 0]
        dy_target = next_pose_target[:, 1] - current_pose[:, 1]
        dyaw_target = self.wrap_angle(next_pose_target[:, 2] - current_pose[:, 2])
        
        target_delta = torch.stack([dx_target, dy_target, dyaw_target], dim=1)
        
        # We can use simple MSE since we target the wrapped angle delta
        loss = nn.functional.mse_loss(delta_pose_pred, target_delta)
        return loss

if __name__ == "__main__":
    print("Testing DynamicsModule...")
    B = 4
    model = DynamicsModule()
    
    pose = torch.randn(B, 3)
    action = torch.randint(0, 16, (B,))
    geometry = torch.randn(B, 64)
    unet_feats = torch.randn(B, 512)
    next_pose = torch.randn(B, 3)
    
    delta = model(pose, action, geometry, unet_feats)
    print(f"Predicted delta shape: {delta.shape}")
    
    loss = model.compute_loss(pose, action, geometry, unet_feats, next_pose)
    print(f"Loss computed successfully: {loss.item():.4f}")
    
    updated = model.apply_update(pose, delta)
    print(f"Updated pose shape: {updated.shape}")
