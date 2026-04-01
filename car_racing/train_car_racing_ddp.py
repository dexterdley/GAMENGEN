import os
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import matplotlib.pyplot as plt

# Enable TensorFloat32 (TF32) for massive speedups on Ampere/Hopper GPUs
torch.set_float32_matmul_precision('high')

### CMD:  torchrun --standalone --nproc_per_node=8 train_car_racing_ddp.py

# ==========================================
# 0. UTILS & HELPERS
# ==========================================
class WorldDataset(Dataset):
    def __init__(self, data_dir="multigen_dataset"):
        self.files = sorted(glob.glob(f"{data_dir}/*.npz"))
        self.data_list = []
        self.n_map_points = 20
        
        # Pre-index all transitions to avoid nested loops during training
        for f in self.files:
            with np.load(f, mmap_mode='r') as data:
                num_steps = data['video'].shape[0] 
                for t in range(num_steps - 1):
                    self.data_list.append((f, t))
        
        print(f"Indexed {len(self.data_list)} transitions from {len(self.files)} files.")
                
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_path, t = self.data_list[idx]
        data = np.load(file_path, mmap_mode='r')
        
        # 1. Player Pose (xt, yt, psi_t)

        pose_t = torch.from_numpy(data['poses'][t]).float()
        car_x, car_y, car_angle = pose_t[0].item(), pose_t[1].item(), pose_t[2].item()
        
        # 2. Visual Context & action (Current Frame)
        img_t = torch.from_numpy(data['video'][t]).permute(2, 0, 1).float() / 255.0
        action_t = torch.from_numpy(data['actions'][t]).float()

        # 3. Static Map M -> Local Geometric Signal
        # We transform the map into the player's EGO-CENTRIC frame
        track = data['track'] # Global vertices (V)
        
        # Find nearest vertices to current pose
        distances = np.linalg.norm(track - np.array([car_x, car_y]), axis=1)
        closest_indices = np.argsort(distances)[:self.n_map_points]
        local_v = track[closest_indices]
        
        # Step A: Translate relative to car
        rel_v = local_v - np.array([car_x, car_y])
        
        # Step B: Rotate by -car_angle so "Forward" is always Y-axis
        # This is the "Ray-traced/Geometric" signal the paper describes
        c, s = np.cos(-car_angle), np.sin(-car_angle)
        rotation_matrix = np.array([[c, -s], [s, c]])
        ego_map = rel_v @ rotation_matrix.T 
        
        # Normalize/Flatten for the fusion layer
        ego_map_tensor = torch.from_numpy(ego_map).float().flatten()

        # 4. Targets (t+1)
        img_next = torch.from_numpy(data['video'][t+1]).permute(2, 0, 1).float() / 255.0
        pose_next = torch.from_numpy(data['poses'][t+1]).float()
        delta_pose = pose_next - pose_t
        
        return img_t, pose_t, action_t, ego_map_tensor, img_next, delta_pose

class WorldModel(nn.Module):
    def __init__(self, state_dim=3, action_dim=3, n_map_points=20):
        super().__init__()
        
        # --- 1. VISUAL ENCODER (The 'Observation' feature extractor) ---
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(),   # 200x300
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),  # 100x150
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(), # 50x75
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(), # 25x37
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten()
        )
        
        # --- 2. MAP & POSE FUSION ---
        # 256 (Visual) + 3 (Pose: x, y, angle) + 3 (Action) + (n_map_points * 2)
        # For 20 points, map_dim = 40.
        self.map_dim = n_map_points * 2
        self.fusion_dim = 256 + state_dim + action_dim + self.map_dim
        
        # --- 3. HEAD A: DYNAMICS MODULE (The Physics Engine) ---
        # Predicts the change in pose based on the persistent state S_t
        self.dynamics_head = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.ReLU(),
            nn.Linear(512, state_dim) # Output: [dx, dy, dangle]
        )
        
        # --- 4. HEAD B: OBSERVATION MODULE (The Neural Renderer) ---
        # We project the fusion vector back into a spatial representation
        self.obs_latents = nn.Linear(self.fusion_dim, 128 * 25 * 37)
        
        self.decoder = nn.Sequential(
            # Start: 128 x 25 x 37
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 50x74
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 100x148
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 200x296
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            
            nn.Upsample(size=(400, 600), mode='bilinear', align_corners=False), # Final Size
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, img, pose, action, local_map):
        # Image Normalization
        if img.max() > 1.0: img = img / 255.0
            
        # Feature Extraction
        visual_features = self.encoder(img) # [Batch, 256]
        
        # --- S_t Construction (Equation 5) ---
        # Concatenate: Vision + Pose + Action + Static Map Reference
        combined = torch.cat([visual_features, pose, action, local_map], dim=1)
        
        # Physics Step
        next_pose_delta = self.dynamics_head(combined)
        
        # Graphics Step (Rendering the high-res hallucination)
        z = self.obs_latents(combined).view(-1, 128, 25, 37)
        next_img = self.decoder(z)
        
        return next_img, next_pose_delta
    
# ==========================================
# 1. DDP INITIALIZATION
# ==========================================
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])

torch.cuda.set_device(local_rank)
DEVICE = torch.device(f"cuda:{local_rank}")

BATCH_SIZE = 128  
EPOCHS = 50
LEARNING_RATE = 3e-4

# ==========================================
# 2. MODEL SETUP
# ==========================================
model = WorldModel(n_map_points=20).to(DEVICE)
model = DDP(model, device_ids=[local_rank])
model = torch.compile(model) 

# ==========================================
# 3. DATASET & DISTRIBUTED SAMPLER
# ==========================================
dataset = WorldDataset("multigen_dataset")

sampler = DistributedSampler(dataset, shuffle=True)

dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    sampler=sampler,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2
)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion_img = torch.nn.L1Loss()
criterion_pose = torch.nn.MSELoss()

# ==========================================
# 4. TRAINING LOOP & PLOT TRACKERS
# ==========================================
if global_rank == 0:
    print(f"Starting Distributed Training on 8 GPUs...")
    # Initialize lists to track the average loss per epoch
    history_img_loss = []
    history_pose_loss = []

for epoch in range(EPOCHS):
    sampler.set_epoch(epoch)
    
    epoch_loss_img = 0
    epoch_loss_pose = 0
    
    if global_rank == 0:
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    else:
        loop = dataloader
        
    for batch in loop:
        img_t, pose_t, action_t, ego_map_t, img_next, target_delta = [
            b.to(DEVICE, non_blocking=True) for b in batch
        ]
        
        # Forward Pass
        pred_img, pred_delta = model(img_t, pose_t, action_t, ego_map_t)
        
        # Calculate Losses
        loss_img = criterion_img(pred_img, img_next)
        loss_pose = criterion_pose(pred_delta, target_delta)
        total_loss = (loss_img * 1.0) + (loss_pose * 1000.0)
        
        # Backward Pass
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        
        # Logging (Only on Rank 0)
        if global_rank == 0:
            epoch_loss_img += loss_img.item()
            epoch_loss_pose += loss_pose.item()
            loop.set_postfix(img_l=loss_img.item(), pose_l=loss_pose.item())

    # ==========================================
    # 5. END OF EPOCH: SAVE CHECKPOINT & PLOT
    # ==========================================
    if global_rank == 0:
        # Calculate epoch averages
        avg_img = epoch_loss_img / len(dataloader)
        avg_pose = epoch_loss_pose / len(dataloader)
        
        history_img_loss.append(avg_img)
        history_pose_loss.append(avg_pose)
        
        # Save Model
        torch.save(model.module.state_dict(), f"./world_models/car_racing_world_model_epoch_{epoch}.pth")
        
        # Generate and save the plot
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Image Loss
        plt.subplot(1, 2, 1)
        plt.plot(history_img_loss, label='Img Loss (L1)', color='blue', marker='o')
        plt.title('Image Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # Subplot 2: Pose Loss
        plt.subplot(1, 2, 2)
        plt.plot(history_pose_loss, label='Pose Loss (MSE)', color='red', marker='o')
        plt.title('Pose Delta Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("loss_curves.png", dpi=150)
        plt.close() # Critical: close the figure to prevent RAM leaks
        
        print(f"Epoch {epoch} finished. Avg Img Loss: {avg_img:.4f} | Avg Pose Loss: {avg_pose:.4f}")

# Cleanup
dist.destroy_process_group()
if global_rank == 0:
    print("Training Complete!")