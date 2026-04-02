import os
import random
import math
import copy
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint as ckpt_fn
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import gym
from stable_baselines3 import PPO
from huggingface_sb3 import load_from_hub
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Enable TensorFloat32 (TF32) for massive speedups on Ampere/Hopper GPUs
torch.set_float32_matmul_precision('high')
np.bool8 = np.bool_

# ==========================================
# 0. ARGUMENT PARSER
# ==========================================
parser = argparse.ArgumentParser(description="Recurrent Diffusion World Model for CarRacing-v2")

# Resolution (common presets: 64x64, 96x96, 128x192, 200x300, 400x600)
parser.add_argument('--img_h', type=int, default=64, help='Image height (default: 64)')
parser.add_argument('--img_w', type=int, default=64, help='Image width (default: 64)')

# Training hyperparameters
parser.add_argument('--batch_size', type=int, default=128, help='Batch size per GPU (default: 128)')
parser.add_argument('--seq_len', type=int, default=8, help='Sequence length for GRU burn-in (default: 8)')
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
parser.add_argument('--steps_per_epoch', type=int, default=200, help='Training steps per epoch (default: 200)')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (default: 3e-4)')
parser.add_argument('--lr_min', type=float, default=1e-6, help='Minimum LR for cosine schedule (default: 1e-6)')
parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')

# Model hyperparameters
parser.add_argument('--diffusion_timesteps', type=int, default=200, help='Diffusion noise schedule steps (default: 200)')
parser.add_argument('--ddim_steps', type=int, default=50, help='DDIM sampling steps at inference (default: 50)')
parser.add_argument('--hidden_dim', type=int, default=256, help='GRU hidden dim (default: 256)')
parser.add_argument('--noise_aug_max', type=int, default=20, help='Max noise augmentation level (default: 20)')
parser.add_argument('--cond_drop_prob', type=float, default=0.1, help='CFG conditioning dropout probability (default: 0.1)')
parser.add_argument('--guidance_scale', type=float, default=1.5, help='CFG guidance scale at inference (default: 1.5)')
parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate (default: 0.999)')

# Data collection
parser.add_argument('--buffer_capacity', type=int, default=5000, help='Replay buffer capacity (default: 5000)')
parser.add_argument('--warmup_steps', type=int, default=3000, help='Warmup rollout steps (default: 3000)')
parser.add_argument('--collect_steps', type=int, default=300, help='Rollout steps per epoch (default: 300)')

# Eval
parser.add_argument('--eval_frames', type=int, default=50, help='Frames to generate during eval (default: 50)')
parser.add_argument('--eval_every', type=int, default=5, help='Run eval every N epochs (default: 5)')
parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs (default: 10)')

args = parser.parse_args()

### Example usage:
### CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 train_car_racing_main.py --img_h 64 --img_w 64 --batch_size 256
### CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 train_car_racing_main.py --img_h 400 --img_w 600 --batch_size 64

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ==========================================
# 1. SEQUENCE REPLAY BUFFER
# ==========================================
class TransitionSequenceBuffer:
    def __init__(self, capacity, img_h, img_w):
        self.capacity = capacity
        self.img_h = img_h
        self.img_w = img_w
        self.ptr = 0
        self.size = 0
        
        self.img_t = np.zeros((capacity, 3, img_h, img_w), dtype=np.uint8)
        self.pose_t = np.zeros((capacity, 3), dtype=np.float32)
        self.action_t = np.zeros((capacity, 3), dtype=np.float32)
        self.ego_map_t = np.zeros((capacity, 40), dtype=np.float32)
        self.img_next = np.zeros((capacity, 3, img_h, img_w), dtype=np.uint8)
        self.pose_next = np.zeros((capacity, 3), dtype=np.float32)
        self.done_t = np.zeros(capacity, dtype=bool)
        
    def push(self, img, pose, action, ego_map, next_img, next_pose, done):
        # Resize if image doesn't match buffer resolution
        h, w = img.shape[:2]
        if h != self.img_h or w != self.img_w:
            img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
            next_img = cv2.resize(next_img, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
        
        self.img_t[self.ptr] = np.transpose(img, (2, 0, 1))
        self.pose_t[self.ptr] = pose
        self.action_t[self.ptr] = action
        self.ego_map_t[self.ptr] = ego_map
        self.img_next[self.ptr] = np.transpose(next_img, (2, 0, 1))
        self.pose_next[self.ptr] = next_pose
        self.done_t[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample_sequence(self, batch_size, seq_len, device):
        """Samples a contiguous sequence of length seq_len."""
        idxs = np.random.randint(0, self.size - seq_len, size=batch_size)
        
        # Ensure sequences don't cross episode boundaries (dones)
        for i in range(batch_size):
            while self.done_t[idxs[i] : idxs[i] + seq_len - 1].any():
                idxs[i] = np.random.randint(0, self.size - seq_len)
                
        i_seq = np.stack([self.img_t[idx : idx + seq_len] for idx in idxs])
        p_seq = np.stack([self.pose_t[idx : idx + seq_len] for idx in idxs])
        a_seq = np.stack([self.action_t[idx : idx + seq_len] for idx in idxs])
        e_seq = np.stack([self.ego_map_t[idx : idx + seq_len] for idx in idxs])
        
        i_next_target = np.stack([self.img_next[idx + seq_len - 1] for idx in idxs])
        p_next_target = np.stack([self.pose_next[idx + seq_len - 1] for idx in idxs])
        
        # Convert to Tensors
        i_seq_t = (torch.from_numpy(i_seq).float().to(device, non_blocking=True) / 127.5) - 1.0
        p_seq_t = torch.from_numpy(p_seq).to(device, non_blocking=True)
        a_seq_t = torch.from_numpy(a_seq).to(device, non_blocking=True)
        e_seq_t = torch.from_numpy(e_seq).to(device, non_blocking=True)
        
        i_next_t = (torch.from_numpy(i_next_target).float().to(device, non_blocking=True) / 127.5) - 1.0
        p_next_t = torch.from_numpy(p_next_target).to(device, non_blocking=True)
        
        delta_p = p_next_t - p_seq_t[:, -1]
        
        return i_seq_t, p_seq_t, a_seq_t, e_seq_t, i_next_t, delta_p

# ==========================================
# 2. COSINE NOISE SCHEDULE
# ==========================================
def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as proposed in Nichol & Dhariwal 2021."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.999)

# ==========================================
# 3. DIFFUSION U-NET WITH ATTENTION & GRU WORLD MODEL
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResBlock(nn.Module):
    """Residual block with GroupNorm and SiLU."""
    def __init__(self, in_ch, out_ch, emb_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.emb_proj(emb).unsqueeze(-1).unsqueeze(-1)
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)

class SelfAttention(nn.Module):
    """Self-attention layer for spatial feature maps."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = self.norm(x).view(b, c, h * w).permute(0, 2, 1)  # (B, HW, C)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        return x + attn_out.permute(0, 2, 1).view(b, c, h, w)

class ConditionalUNet(nn.Module):
    """Resolution-agnostic U-Net with self-attention at the 3rd downsampled level."""
    def __init__(self, cond_dim=256, base_ch=32):
        super().__init__()
        # Channel progression: base_ch -> 2x -> 4x -> 8x -> 16x
        ch1, ch2, ch3, ch4, ch5 = base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(64),
            nn.Linear(64, 256), nn.GELU(),
            nn.Linear(256, 256)
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, 256), nn.GELU(),
            nn.Linear(256, 256)
        )
        
        # Noise augmentation level embedding (used during training)
        self.noise_aug_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(64),
            nn.Linear(64, 256), nn.GELU(),
            nn.Linear(256, 256)
        )
        
        # Encoder path (4 downsamples via MaxPool2d(2))
        self.inc = ResBlock(6, ch1, emb_dim=256)
        
        self.down1 = nn.MaxPool2d(2)
        self.res_down1 = ResBlock(ch1, ch2, emb_dim=256)
        
        self.down2 = nn.MaxPool2d(2)
        self.res_down2 = ResBlock(ch2, ch3, emb_dim=256)
        
        self.down3 = nn.MaxPool2d(2)
        self.res_down3 = ResBlock(ch3, ch4, emb_dim=256)
        self.attn_down3 = SelfAttention(ch4)
        
        self.down4 = nn.MaxPool2d(2)
        self.res_down4 = ResBlock(ch4, ch5, emb_dim=256)
        
        # Bottleneck
        self.bot1 = ResBlock(ch5, ch5, emb_dim=256)
        self.bot2 = ResBlock(ch5, ch5, emb_dim=256)
        
        # Decoder path (4 upsamples via F.interpolate)
        self.res_up1 = ResBlock(ch5 + ch4, ch4, emb_dim=256)
        self.attn_up1 = SelfAttention(ch4)
        
        self.res_up2 = ResBlock(ch4 + ch3, ch3, emb_dim=256)
        self.res_up3 = ResBlock(ch3 + ch2, ch2, emb_dim=256)
        self.res_up4 = ResBlock(ch2 + ch1, ch1, emb_dim=256)
        
        self.outc = nn.Conv2d(ch1, 3, kernel_size=3, padding=1)

    def _encoder_block(self, x, emb):
        x1 = self.inc(x, emb)
        x2 = self.res_down1(self.down1(x1), emb)
        x3 = self.res_down2(self.down2(x2), emb)
        x4 = self.attn_down3(self.res_down3(self.down3(x3), emb))
        x5 = self.res_down4(self.down4(x4), emb)
        return x1, x2, x3, x4, x5
    
    def _decoder_block(self, x5, x4, x3, x2, x1, emb):
        x5 = self.bot1(x5, emb)
        x5 = self.bot2(x5, emb)
        
        up5 = F.interpolate(x5, size=x4.shape[2:], mode='nearest')
        x = self.attn_up1(self.res_up1(torch.cat([up5, x4], dim=1), emb))
        
        up4 = F.interpolate(x, size=x3.shape[2:], mode='nearest')
        x = self.res_up2(torch.cat([up4, x3], dim=1), emb)
        
        up3 = F.interpolate(x, size=x2.shape[2:], mode='nearest')
        x = self.res_up3(torch.cat([up3, x2], dim=1), emb)
        
        up2 = F.interpolate(x, size=x1.shape[2:], mode='nearest')
        x = self.res_up4(torch.cat([up2, x1], dim=1), emb)
        
        return self.outc(x)

    def forward(self, x_noisy, t, x_cond, h_t, noise_aug_level=None):
        emb = self.time_mlp(t) + self.cond_mlp(h_t)
        
        if noise_aug_level is not None:
            emb = emb + self.noise_aug_mlp(noise_aug_level.float())
        
        x = torch.cat([x_noisy, x_cond], dim=1)
        
        if self.training:
            x1, x2, x3, x4, x5 = ckpt_fn(self._encoder_block, x, emb, use_reentrant=False)
            return ckpt_fn(self._decoder_block, x5, x4, x3, x2, x1, emb, use_reentrant=False)
        else:
            x1, x2, x3, x4, x5 = self._encoder_block(x, emb)
            return self._decoder_block(x5, x4, x3, x2, x1, emb)

class RecurrentDiffusionWorldModel(nn.Module):
    def __init__(self, timesteps=200, hidden_dim=256, noise_aug_max=20, cond_drop_prob=0.1, base_ch=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.noise_aug_max = noise_aug_max
        self.cond_drop_prob = cond_drop_prob
        self.unet = ConditionalUNet(cond_dim=hidden_dim, base_ch=base_ch)
        
        # Vision Encoder: adaptive depth via AdaptiveAvgPool2d
        # Works for any resolution, always outputs enc_dim features
        enc_dim = 128
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, enc_dim, 4, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        
        # Memory Module (RNN)
        # Input: enc_dim(img) + 3(pose) + 3(act) + 40(map)
        gru_input_dim = enc_dim + 3 + 3 + 40
        self.gru = nn.GRUCell(gru_input_dim, hidden_dim)
        
        # Pose prediction head
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(),
            nn.Linear(128, 3)
        )
        
        # Cosine noise schedule (better than linear)
        self.timesteps = timesteps
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        # For posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer('posterior_variance', betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))

    def forward(self, img_seq, pose_seq, act_seq, ego_seq, img_next_target=None):
        b, seq_len, _, _, _ = img_seq.shape
        device = img_seq.device
        
        # Initialize hidden state
        h = torch.zeros(b, self.hidden_dim, device=device)
        
        # Burn-in: Unroll the GRU over the sequence to build momentum
        for t in range(seq_len):
            z_t = self.encoder(img_seq[:, t])
            vec_in = torch.cat([z_t, pose_seq[:, t]/100.0, act_seq[:, t], ego_seq[:, t]/50.0], dim=1)
            h = self.gru(vec_in, h)
            
        # h now contains the physical momentum up to the final step.
        pred_pose_delta = self.pose_head(h)
        
        if img_next_target is not None:
            t_diff = torch.randint(0, self.timesteps, (b,), device=device).long()
            noise = torch.randn_like(img_next_target)
            
            sqrt_alpha_t = self.sqrt_alphas_cumprod[t_diff].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t_diff].view(-1, 1, 1, 1)
            noisy_img = sqrt_alpha_t * img_next_target + sqrt_one_minus_alpha_t * noise
            
            # --- NOISE AUGMENTATION (GameNGen technique) ---
            # During training, corrupt the conditioning image with small amounts of noise
            # to simulate the imperfect outputs the model sees during autoregressive inference.
            x_cond = img_seq[:, -1]
            noise_aug_level = torch.zeros(b, device=device, dtype=torch.long)
            h_unet = h.clone()
            
            if self.training:
                noise_aug_level = torch.randint(0, self.noise_aug_max, (b,), device=device).long()
                aug_noise = torch.randn_like(x_cond)
                sqrt_alpha_aug = self.sqrt_alphas_cumprod[noise_aug_level].view(-1, 1, 1, 1)
                sqrt_one_minus_aug = self.sqrt_one_minus_alphas_cumprod[noise_aug_level].view(-1, 1, 1, 1)
                x_cond = sqrt_alpha_aug * x_cond + sqrt_one_minus_aug * aug_noise
                
                # --- CLASSIFIER-FREE GUIDANCE: Random conditioning dropout ---
                # Zero out both h and x_cond for a fraction of samples so the
                # model learns an unconditional noise prediction.
                drop_mask = torch.rand(b, device=device) < self.cond_drop_prob
                if drop_mask.any():
                    h_unet[drop_mask] = 0.0
                    x_cond[drop_mask] = 0.0
            
            pred_noise = self.unet(noisy_img, t_diff, x_cond, h_unet, noise_aug_level=noise_aug_level)
            return pred_noise, noise, pred_pose_delta
        
        return pred_pose_delta

    @torch.no_grad()
    def sample_autoregressive(self, img_t, pose_t, action_t, ego_map_t, h_prev, 
                              ddim_steps=50, guidance_scale=1.5):
        """DDIM sampling with Classifier-Free Guidance for sharper outputs."""
        b, c, h_dim, w_dim = img_t.shape
        device = img_t.device
        use_cfg = guidance_scale > 1.0
        
        # 1. Update Recurrent Memory
        z_t = self.encoder(img_t)
        vec_in = torch.cat([z_t, pose_t/100.0, action_t, ego_map_t/50.0], dim=1)
        h_curr = self.gru(vec_in, h_prev)
        
        # 2. DDIM Denoising Loop
        step_size = self.timesteps // ddim_steps
        ddim_timesteps = list(range(0, self.timesteps, step_size))
        ddim_timesteps_prev = [-1] + ddim_timesteps[:-1]
        
        x = torch.randn((b, 3, h_dim, w_dim), device=device)
        
        # At inference, no noise augmentation -> level = 0
        noise_aug_level = torch.zeros(b, device=device, dtype=torch.long)
        
        # Prepare unconditional inputs for CFG
        if use_cfg:
            h_uncond = torch.zeros_like(h_curr)
            img_uncond = torch.zeros_like(img_t)
        
        for i in reversed(range(len(ddim_timesteps))):
            t_cur = ddim_timesteps[i]
            t_prev = ddim_timesteps_prev[i]
            
            t_batch = torch.full((b,), t_cur, device=device, dtype=torch.long)
            
            if use_cfg:
                # --- CLASSIFIER-FREE GUIDANCE ---
                # Run UNet twice: once conditional, once unconditional
                # Then blend: pred = uncond + scale * (cond - uncond)
                x_double = torch.cat([x, x], dim=0)
                t_double = torch.cat([t_batch, t_batch], dim=0)
                img_double = torch.cat([img_t, img_uncond], dim=0)
                h_double = torch.cat([h_curr, h_uncond], dim=0)
                aug_double = torch.cat([noise_aug_level, noise_aug_level], dim=0)
                
                pred_noise_both = self.unet(x_double, t_double, img_double, h_double, noise_aug_level=aug_double)
                pred_noise_cond, pred_noise_uncond = pred_noise_both.chunk(2, dim=0)
                pred_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)
            else:
                pred_noise = self.unet(x, t_batch, img_t, h_curr, noise_aug_level=noise_aug_level)
            
            alpha_cumprod_t = self.alphas_cumprod[t_cur]
            alpha_cumprod_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
            
            # DDIM deterministic step (eta=0)
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            dir_xt = torch.sqrt(1 - alpha_cumprod_prev) * pred_noise
            x = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt

        pred_pose_delta = self.pose_head(h_curr)
        return torch.clamp(x, -1.0, 1.0), pred_pose_delta, h_curr

# ==========================================
# 4. EMA (Exponential Moving Average)
# ==========================================
class EMAModel:
    """Maintains EMA weights for stable inference."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    def forward(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)

# ==========================================
# 5. SETUP & INITIALIZATION
# ==========================================
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])
torch.cuda.set_device(local_rank)
DEVICE = torch.device(f"cuda:{local_rank}")

IMG_H, IMG_W = args.img_h, args.img_w
BATCH_SIZE = args.batch_size
SEQ_LEN = args.seq_len
EPOCHS = args.epochs
STEPS_PER_EPOCH = args.steps_per_epoch
COLLECT_STEPS = args.collect_steps
LEARNING_RATE = args.lr
DIFFUSION_TIMESTEPS = args.diffusion_timesteps
DDIM_INFERENCE_STEPS = args.ddim_steps

set_seed(args.seed + global_rank)

if global_rank == 0:
    print(f"Config: {IMG_H}x{IMG_W} | BS={BATCH_SIZE} | SeqLen={SEQ_LEN} | LR={LEARNING_RATE}")
    print(f"        Epochs={EPOCHS} | Steps/Epoch={STEPS_PER_EPOCH} | Diffusion T={DIFFUSION_TIMESTEPS}")

if global_rank == 0:
    ckpt_path = load_from_hub(repo_id="igpaub/ppo-CarRacing-v2", filename="ppo-CarRacing-v2.zip")
dist.barrier()
if global_rank != 0:
    ckpt_path = load_from_hub(repo_id="igpaub/ppo-CarRacing-v2", filename="ppo-CarRacing-v2.zip")

# Determine base UNet channels: smaller for high-res to save VRAM
base_ch = 32 if (IMG_H * IMG_W) > 64 * 64 else 64

agent = PPO.load(ckpt_path, device=DEVICE)
env = gym.make("CarRacing-v2", continuous=True, render_mode="rgb_array")
buffer = TransitionSequenceBuffer(capacity=args.buffer_capacity, img_h=IMG_H, img_w=IMG_W)

model = RecurrentDiffusionWorldModel(
    timesteps=DIFFUSION_TIMESTEPS,
    hidden_dim=args.hidden_dim,
    noise_aug_max=args.noise_aug_max,
    cond_drop_prob=args.cond_drop_prob,
    base_ch=base_ch
).to(DEVICE)
model = DDP(model, device_ids=[local_rank])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler('cuda')

# Cosine LR schedule
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS * STEPS_PER_EPOCH, eta_min=args.lr_min
)

# EMA for stable inference (only on rank 0)
if global_rank == 0:
    ema = EMAModel(model.module, decay=args.ema_decay)

criterion_noise = torch.nn.MSELoss()
criterion_pose = torch.nn.MSELoss()

# ==========================================
# 6. DATA COLLECTION HELPER
# ==========================================
def collect_rollouts(env, agent, buffer, n_steps):
    obs, _ = env.reset()
    for _ in range(50):
        obs, _, _, _, _ = env.step([0.0, 0.0, 0.0])
        
    track_vertices = np.array([[t[2], t[3]] for t in env.unwrapped.track])
    
    for _ in tqdm(range(n_steps), disable=global_rank!=0):
        action, _states = agent.predict(obs, deterministic=True)
        action_floats = [float(a) for a in action]
        
        img_t = env.render()
        car_x, car_y = env.unwrapped.car.hull.position
        car_angle = env.unwrapped.car.hull.angle
        pose_t = np.array([car_x, car_y, car_angle], dtype=np.float32)
        
        distances = np.linalg.norm(track_vertices - np.array([car_x, car_y]), axis=1)
        closest_indices = np.argsort(distances)[:20]
        local_v = track_vertices[closest_indices]
        rel_v = local_v - np.array([car_x, car_y])
        c, s = np.cos(-car_angle), np.sin(-car_angle)
        rotation_matrix = np.array([[c, -s], [s, c]])
        ego_map_t = (rel_v @ rotation_matrix.T).flatten().astype(np.float32)
        
        next_obs, reward, terminated, truncated, _ = env.step(action_floats)
        done = terminated or truncated
        
        img_next = env.render()
        next_x, next_y = env.unwrapped.car.hull.position
        next_angle = env.unwrapped.car.hull.angle
        pose_next = np.array([next_x, next_y, next_angle], dtype=np.float32)
        
        buffer.push(img_t, pose_t, np.array(action_floats, dtype=np.float32), ego_map_t, img_next, pose_next, done)
        
        obs = next_obs
        if done:
            obs, _ = env.reset()
            for _ in range(50): obs, _, _, _, _ = env.step([0.0, 0.0, 0.0])
            track_vertices = np.array([[t[2], t[3]] for t in env.unwrapped.track])

# ==========================================
# 7. AUTOREGRESSIVE EVALUATION LOOP (uses EMA model)
# ==========================================
def generate_autoregressive_video(ema_model, env, agent, device, num_frames=100, ddim_steps=100): 
    ema_model.eval()
    val_img_losses = []
    val_pose_losses = []
    
    with torch.no_grad():
        obs, _ = env.reset()
        for _ in range(50):
            obs, _, _, _, _ = env.step([0.0, 0.0, 0.0])
            
        track_vertices = np.array([[t[2], t[3]] for t in env.unwrapped.track])
        img = env.render()
        
        car_x, car_y = env.unwrapped.car.hull.position
        car_angle = env.unwrapped.car.hull.angle
        pose = np.array([car_x, car_y, car_angle], dtype=np.float32)
        
        c, s = np.cos(-car_angle), np.sin(-car_angle)
        rot = np.array([[c, -s], [s, c]])
        dists = np.linalg.norm(track_vertices - pose[:2], axis=1)
        closest = np.argsort(dists)[:20]
        ego_map = ((track_vertices[closest] - pose[:2]) @ rot.T).flatten().astype(np.float32)
        
        pred_videos = []
        true_videos = []
        
        # Resize initial frame to match training resolution
        img_resized = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA) if (img.shape[0] != IMG_H or img.shape[1] != IMG_W) else img
        curr_img_t = (torch.from_numpy(np.transpose(img_resized, (2, 0, 1))).float().to(device).unsqueeze(0) / 127.5) - 1.0
        curr_pose_t = torch.from_numpy(pose).to(device).unsqueeze(0)
        curr_ego_map_t = torch.from_numpy(ego_map).to(device).unsqueeze(0)
        
        h_curr = torch.zeros(1, ema_model.hidden_dim, device=device)
        
        for t in tqdm(range(num_frames), desc="Dreaming (DDIM + RNN)"):
            action, _ = agent.predict(obs, deterministic=True)
            action_floats = [float(a) for a in action]
            act_t = torch.tensor(action_floats, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Use EMA model for inference with DDIM sampling
            next_img_pred, next_pose_delta, h_curr = ema_model.sample_autoregressive(
                curr_img_t, curr_pose_t, act_t, curr_ego_map_t, h_curr, ddim_steps=ddim_steps
            )
            
            pred_img_np = ((next_img_pred.squeeze(0).cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            pred_videos.append(pred_img_np)
            
            obs, _, terminated, truncated, _ = env.step(action_floats)
            true_img = env.render()
            # Resize ground truth to match training resolution for val loss
            true_img_resized = cv2.resize(true_img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA) if (true_img.shape[0] != IMG_H or true_img.shape[1] != IMG_W) else true_img
            true_videos.append(true_img_resized)
            
            # --- Val Loss: compare predicted frame vs ground-truth frame ---
            true_img_t = (torch.from_numpy(np.transpose(true_img_resized, (2, 0, 1))).float().to(device).unsqueeze(0) / 127.5) - 1.0
            val_img_losses.append(F.l1_loss(next_img_pred, true_img_t).item())
            
            # --- Val Loss: compare predicted pose delta vs ground-truth pose delta ---
            gt_x, gt_y = env.unwrapped.car.hull.position
            gt_angle = env.unwrapped.car.hull.angle
            gt_pose_t = torch.tensor([[gt_x, gt_y, gt_angle]], dtype=torch.float32, device=device)
            gt_delta = gt_pose_t - (curr_pose_t - next_pose_delta)  # delta from pre-update pose
            val_pose_losses.append(F.mse_loss(next_pose_delta, gt_delta).item())
            
            if terminated or truncated:
                break
                
            curr_img_t = next_img_pred
            curr_pose_t = curr_pose_t + next_pose_delta
            
            pred_pose_np = curr_pose_t.squeeze(0).cpu().numpy()
            px, py, pang = pred_pose_np
            c_, s_ = np.cos(-pang), np.sin(-pang)
            rot_ = np.array([[c_, -s_], [s_, c_]])
            dists_ = np.linalg.norm(track_vertices - np.array([px, py]), axis=1)
            closest_ = np.argsort(dists_)[:20]
            new_ego = ((track_vertices[closest_] - np.array([px, py])) @ rot_.T).flatten().astype(np.float32)
            curr_ego_map_t = torch.from_numpy(new_ego).to(device).unsqueeze(0)
    
    avg_val_img = np.mean(val_img_losses) if val_img_losses else 0.0
    avg_val_pose = np.mean(val_pose_losses) if val_pose_losses else 0.0
    return np.array(true_videos), np.array(pred_videos), avg_val_img, avg_val_pose

# ==========================================
# 8. MAIN TRAINING LOOP
# ==========================================
WARMUP_STEPS = args.warmup_steps
if global_rank == 0: print(f"Warming up buffers: Collecting {WARMUP_STEPS} initial transitions per GPU...")
collect_rollouts(env, agent, buffer, n_steps=WARMUP_STEPS)
dist.barrier()

if global_rank == 0:
    print("Warmup complete! Starting Recurrent Diffusion Training (v2: Noise Aug + DDIM + Attention + EMA)...")
    history_noise_loss, history_pose_loss = [], []
    experiment_name = "experiment_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"./runs/{experiment_name}")

for epoch in range(EPOCHS):
    if global_rank == 0: print(f"Epoch {epoch+1}: Collecting {COLLECT_STEPS} new transitions per GPU...")
    collect_rollouts(env, agent, buffer, n_steps=COLLECT_STEPS)
    dist.barrier() 
    
    epoch_loss_noise = 0
    epoch_loss_pose = 0
    
    loop = range(STEPS_PER_EPOCH)
    if global_rank == 0:
        loop = tqdm(loop, desc=f"Epoch {epoch+1}/{EPOCHS} Training")
        
    for step in loop:
        img_seq, pose_seq, act_seq, ego_seq, next_img_target, target_delta = buffer.sample_sequence(BATCH_SIZE, SEQ_LEN, DEVICE)
        
        # Mixed precision forward pass
        with autocast('cuda', dtype=torch.bfloat16):
            pred_noise, true_noise, pred_delta = model(img_seq, pose_seq, act_seq, ego_seq, img_next_target=next_img_target)
            loss_noise = criterion_noise(pred_noise, true_noise)
            loss_pose = criterion_pose(pred_delta, target_delta)
            total_loss = loss_noise + (loss_pose * 10.0) 
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Update EMA weights
        if global_rank == 0:
            ema.update(model.module)
        
        if global_rank == 0:
            epoch_loss_noise += loss_noise.item()
            epoch_loss_pose += loss_pose.item()
            loop.set_postfix(noise_l=loss_noise.item(), pose_l=loss_pose.item(), lr=scheduler.get_last_lr()[0])

    if global_rank == 0:
        avg_noise = epoch_loss_noise / STEPS_PER_EPOCH
        avg_pose = epoch_loss_pose / STEPS_PER_EPOCH
        history_noise_loss.append(avg_noise)
        history_pose_loss.append(avg_pose)
        
        writer.add_scalar("Loss/Diffusion_Noise", avg_noise, epoch)
        writer.add_scalar("Loss/Pose", avg_pose, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        if (epoch + 1) % args.eval_every == 0 or epoch == EPOCHS - 1:
            print("Generating autoregressive diffusion video (EMA + DDIM) for TensorBoard...")
            true_vid, pred_vid, val_img_loss, val_pose_loss = generate_autoregressive_video(
                ema.shadow, env, agent, DEVICE, num_frames=args.eval_frames, ddim_steps=DDIM_INFERENCE_STEPS
            )
            combined_vid = np.concatenate([true_vid, pred_vid], axis=2)
            vid_tensor = torch.from_numpy(combined_vid).permute(0, 3, 1, 2).unsqueeze(0)
            writer.add_video("Autoregressive/True_Left_Pred_Right", vid_tensor, epoch, fps=15)
            
            # Log autoregressive validation losses
            writer.add_scalar("ValLoss/Image_L1", val_img_loss, epoch)
            writer.add_scalar("ValLoss/Pose_MSE", val_pose_loss, epoch)
            print(f"  Val Image L1: {val_img_loss:.4f} | Val Pose MSE: {val_pose_loss:.4f}")
        
        if (epoch + 1) % args.save_every == 0:
            os.makedirs("./world_models", exist_ok=True)
            # Save both normal and EMA weights
            torch.save(model.module.state_dict(), f"./world_models/car_racing_diffusion_rnn_epoch_{epoch}.pth")
            torch.save(ema.shadow.state_dict(), f"./world_models/car_racing_diffusion_rnn_ema_epoch_{epoch}.pth")
            
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history_noise_loss, label='Noise Loss (MSE)', color='blue', marker='o')
        plt.title('Diffusion Denoising Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history_pose_loss, label='Pose Loss (MSE)', color='red', marker='o')
        plt.title('Pose Delta Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("loss_curves.png", dpi=150)
        plt.close()

dist.destroy_process_group()
if global_rank == 0:
    writer.close()
    print("Training Complete!")