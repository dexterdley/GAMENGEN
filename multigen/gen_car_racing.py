"""
gen_car_racing.py — MultiGen applied to Gymnasium CarRacing-v2

Applies the MultiGen "explicit memory" architecture to the 2D top-down
CarRacing environment. Because the world is 2D, we skip 3D ray-tracing
and instead extract track geometry + car pose directly from Box2D internals.

Architecture:
  Memory Module  → Track polygons (M) + car pose (x, y, θ)
  Observation Module → Diffusion UNet conditioned on geometry mask + action + context
  Dynamics Module → Small MLP predicting Δ(x, y, θ) from action + geometry

Usage:
    python gen_car_racing.py                     # Train
    python gen_car_racing.py --mode infer         # Inference
    python gen_car_racing.py --mode collect_only  # Just collect data
"""

import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm

import gymnasium as gym
from diffusers import DDPMScheduler, UNet2DConditionModel

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
IMG_SIZE = 96           # CarRacing native is 96x96
CONTEXT_FRAMES = 4      # L past frames
ACTION_DIM = 5          # Discretized: [noop, left, right, gas, brake]
CONTEXT_NOISE_SCALE = 0.1
DYNAMICS_LOSS_WEIGHT = 0.1

REPLAY_BUFFER_SIZE = 20000
MIN_BUFFER_SIZE = 512
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
EPISODES = 100
LOG_EVERY = 20
SAVE_EVERY = 500
NUM_TRAIN_TIMESTEPS = 1000
NUM_INFERENCE_STEPS = 20


# ═══════════════════════════════════════════════════════════════════
# MEMORY MODULE — Extract track + pose from Box2D internals
# ═══════════════════════════════════════════════════════════════════
class CarRacingMemory:
    """
    Explicit external memory for CarRacing.
    - Map M: track polygon coordinates from env.unwrapped.track
    - Pose p_t: (x, y, θ) from env.unwrapped.car.hull
    """

    def __init__(self, env):
        self.env = env

    def get_track(self):
        """
        Returns track as list of (beta, x, y, ...) tuples.
        Each entry is a waypoint along the track centerline.
        """
        track = getattr(self.env.unwrapped, 'track', None)
        if track is None:
            return []
        return track

    def get_pose(self):
        """Returns (x, y, angle) of the car."""
        car = getattr(self.env.unwrapped, 'car', None)
        if car is None:
            return (0.0, 0.0, 0.0)
        hull = car.hull
        x, y = hull.position.x, hull.position.y
        angle = hull.angle
        return (float(x), float(y), float(angle))

    def get_velocity(self):
        """Returns (vx, vy, angular_vel) of the car."""
        car = getattr(self.env.unwrapped, 'car', None)
        if car is None:
            return (0.0, 0.0, 0.0)
        hull = car.hull
        vx, vy = hull.linearVelocity.x, hull.linearVelocity.y
        av = hull.angularVelocity
        return (float(vx), float(vy), float(av))

    def render_geometry_mask(self, img_size=IMG_SIZE):
        """
        Draws the track + car position onto a binary mask.
        This is the geometric conditioning signal r_t (analogous to disparity in MultiGen).

        Returns:
            Tensor (1, H, W) float32 in [0, 1]
        """
        import cv2

        mask = np.zeros((img_size, img_size), dtype=np.float32)
        track = self.get_track()
        pose = self.get_pose()

        if not track:
            return torch.zeros(1, img_size, img_size)

        # Gather all track x, y coords
        track_xs = [t[2] for t in track]  # x
        track_ys = [t[3] for t in track]  # y

        # Compute bounding box for normalization
        all_xs = track_xs + [pose[0]]
        all_ys = track_ys + [pose[1]]
        x_min, x_max = min(all_xs) - 10, max(all_xs) + 10
        y_min, y_max = min(all_ys) - 10, max(all_ys) + 10

        def to_pixel(x, y):
            px = int((x - x_min) / (x_max - x_min + 1e-6) * (img_size - 1))
            py = int((y - y_min) / (y_max - y_min + 1e-6) * (img_size - 1))
            py = img_size - 1 - py  # Flip y
            return (np.clip(px, 0, img_size - 1), np.clip(py, 0, img_size - 1))

        # Draw track as connected line
        pts = [to_pixel(x, y) for x, y in zip(track_xs, track_ys)]
        for i in range(len(pts) - 1):
            cv2.line(mask, pts[i], pts[i + 1], 0.7, thickness=2)
        # Close the loop
        if len(pts) > 1:
            cv2.line(mask, pts[-1], pts[0], 0.7, thickness=2)

        # Draw car position
        car_px = to_pixel(pose[0], pose[1])
        cv2.circle(mask, car_px, 3, 1.0, -1)

        # Draw heading arrow
        arrow_len = 5
        end_x = car_px[0] + int(arrow_len * math.cos(pose[2]))
        end_y = car_px[1] - int(arrow_len * math.sin(pose[2]))
        cv2.arrowedLine(mask, car_px,
                        (np.clip(end_x, 0, img_size - 1),
                         np.clip(end_y, 0, img_size - 1)),
                        1.0, thickness=1)

        return torch.tensor(mask, dtype=torch.float32).unsqueeze(0)


# ═══════════════════════════════════════════════════════════════════
# DISCRETIZE ACTIONS — CarRacing uses continuous [steer, gas, brake]
# ═══════════════════════════════════════════════════════════════════
DISCRETE_ACTIONS = [
    np.array([0.0, 0.0, 0.0]),    # 0: noop
    np.array([-1.0, 0.0, 0.0]),   # 1: left
    np.array([1.0, 0.0, 0.0]),    # 2: right
    np.array([0.0, 1.0, 0.0]),    # 3: gas
    np.array([0.0, 0.0, 0.8]),    # 4: brake
]


def random_action():
    """Weighted random policy: favors gas."""
    weights = [0.05, 0.15, 0.15, 0.55, 0.10]
    return np.random.choice(len(DISCRETE_ACTIONS), p=weights)


# ═══════════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════════
class Transition:
    __slots__ = ['target_frame', 'context_frames', 'geometry',
                 'action', 'pose_current', 'pose_next']

    def __init__(self, target_frame, context_frames, geometry,
                 action, pose_current, pose_next):
        self.target_frame = target_frame
        self.context_frames = context_frames
        self.geometry = geometry
        self.action = action
        self.pose_current = pose_current
        self.pose_next = pose_next


class ReplayBuffer:
    def __init__(self, capacity=REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, t): self.buffer.append(t)
    def __len__(self): return len(self.buffer)

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        return {
            'target_frame':   torch.stack([t.target_frame for t in batch]),
            'context_frames': torch.stack([t.context_frames for t in batch]),
            'geometry':        torch.stack([t.geometry for t in batch]),
            'action':          torch.tensor([t.action for t in batch], dtype=torch.long),
            'pose_current':    torch.stack([t.pose_current for t in batch]),
            'pose_next':       torch.stack([t.pose_next for t in batch]),
        }


class ContextTracker:
    def __init__(self, L=CONTEXT_FRAMES, size=IMG_SIZE):
        self.L = L
        self.buffer = torch.zeros(3 * L, size, size)

    def reset(self): self.buffer.zero_()

    def push(self, frame):
        self.buffer = torch.roll(self.buffer, -3, 0)
        self.buffer[-3:] = frame.detach().clone()

    def get(self): return self.buffer.clone()


def frame_to_tensor(obs):
    """CarRacing obs (96, 96, 3) uint8 → (3, 96, 96) float [-1, 1]."""
    t = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1) / 255.0
    return t * 2.0 - 1.0


# ═══════════════════════════════════════════════════════════════════
# OBSERVATION MODULE — Diffusion UNet for next-frame prediction
# ═══════════════════════════════════════════════════════════════════
class ObservationModule(nn.Module):
    """
    Predicts the next frame via diffusion, conditioned on:
    - L context frames (channel concat)
    - Geometry mask from Memory (channel concat)
    - Discrete action (cross-attention)
    """
    def __init__(self, context_frames=CONTEXT_FRAMES, action_dim=ACTION_DIM):
        super().__init__()
        # Input: [noised_target(3)] + [context(3*L)] + [geometry(1)]
        in_ch = 3 + 3 * context_frames + 1

        self.unet = UNet2DConditionModel(
            sample_size=IMG_SIZE,
            in_channels=in_ch,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 256),
            down_block_types=(
                "CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D", "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D", "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
            ),
            cross_attention_dim=action_dim,
        )

        self.action_embed = nn.Embedding(action_dim, action_dim)
        self.scheduler = DDPMScheduler(
            num_train_timesteps=NUM_TRAIN_TIMESTEPS,
            prediction_type="v_prediction",
        )

    def forward(self, x_noisy, timesteps, context, geometry, actions):
        act_emb = self.action_embed(actions).unsqueeze(1)  # (B,1,D)
        inp = torch.cat([x_noisy, context, geometry], dim=1)
        return self.unet(inp, timesteps, encoder_hidden_states=act_emb,
                         return_dict=False)[0]

    def compute_loss(self, target, context, geometry, actions,
                     noise_scale=CONTEXT_NOISE_SCALE):
        B = target.shape[0]
        device = target.device

        t = torch.randint(0, NUM_TRAIN_TIMESTEPS, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(target)
        x_noisy = self.scheduler.add_noise(target, noise, t)
        v_target = self.scheduler.get_velocity(target, noise, t)

        # Noised-context augmentation (§3.2.4)
        if self.training and noise_scale > 0:
            ns = torch.rand(1, device=device) * noise_scale
            context = torch.clamp(
                context + torch.randn_like(context) * ns.view(-1, 1, 1, 1),
                -1, 1)

        v_pred = self.forward(x_noisy, t, context, geometry, actions)
        return F.mse_loss(v_pred, v_target)

    def get_mid_features(self, x_noisy, timesteps, context, geometry, actions):
        """Extract pooled UNet mid-block features (detached) for Dynamics."""
        act_emb = self.action_embed(actions).unsqueeze(1)
        inp = torch.cat([x_noisy, context, geometry], dim=1)

        t_emb = self.unet.time_proj(timesteps).to(dtype=self.unet.dtype)
        emb = self.unet.time_embedding(t_emb)
        sample = self.unet.conv_in(inp)

        for blk in self.unet.down_blocks:
            if hasattr(blk, "has_cross_attention") and blk.has_cross_attention:
                sample, _ = blk(hidden_states=sample, temb=emb,
                                encoder_hidden_states=act_emb)
            else:
                sample, _ = blk(hidden_states=sample, temb=emb)

        sample = self.unet.mid_block(sample, temb=emb,
                                      encoder_hidden_states=act_emb)
        pooled = F.adaptive_avg_pool2d(sample, (1, 1)).view(B := x_noisy.shape[0], -1)
        return pooled.detach()


# ═══════════════════════════════════════════════════════════════════
# DYNAMICS MODULE — Tiny MLP for 2D car physics
# ═══════════════════════════════════════════════════════════════════
class DynamicsModule(nn.Module):
    """
    Predicts Δ(x, y, θ) from (pose, action, geometry_summary, unet_features).
    Simplified MLP since car physics is 2D.
    """
    def __init__(self, pose_dim=3, action_dim=ACTION_DIM,
                 geom_summary_dim=32, unet_feat_dim=256):
        super().__init__()
        self.action_embed = nn.Embedding(action_dim, 16)
        self.geom_proj = nn.Linear(IMG_SIZE, geom_summary_dim)

        total_in = pose_dim + 16 + geom_summary_dim + unet_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_in, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 3),  # Δx, Δy, Δθ
        )

    def forward(self, pose, action, geom_1d, unet_feats):
        a = self.action_embed(action)
        g = self.geom_proj(geom_1d)
        x = torch.cat([pose, a, g, unet_feats], dim=1)
        return self.mlp(x)

    def compute_loss(self, pose_curr, action, geom_1d, unet_feats, pose_next):
        delta_pred = self.forward(pose_curr, action, geom_1d, unet_feats)
        dx = pose_next[:, 0] - pose_curr[:, 0]
        dy = pose_next[:, 1] - pose_curr[:, 1]
        da = (pose_next[:, 2] - pose_curr[:, 2] + math.pi) % (2 * math.pi) - math.pi
        delta_gt = torch.stack([dx, dy, da], dim=1)
        return F.mse_loss(delta_pred, delta_gt)


# ═══════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════
def train(args):
    device = args.device
    env = gym.make("CarRacing-v2", render_mode="rgb_array")

    memory = CarRacingMemory(env)
    obs_net = ObservationModule().to(device)
    dyn_net = DynamicsModule(unet_feat_dim=256).to(device)  # mid-block channels

    optimizer = torch.optim.Adam(
        list(obs_net.parameters()) + list(dyn_net.parameters()),
        lr=LEARNING_RATE)

    replay = ReplayBuffer()
    ctx = ContextTracker()

    obs_net.train()
    dyn_net.train()
    os.makedirs(args.save_dir, exist_ok=True)

    steps = 0
    cum_obs, cum_dyn = 0.0, 0.0

    for ep in tqdm(range(args.episodes), desc="Episodes"):
        obs, info = env.reset()
        ctx.reset()
        ctx.push(frame_to_tensor(obs))

        done = False
        while not done:
            # --- Collect ---
            pose_curr = torch.tensor(memory.get_pose(), dtype=torch.float32)
            geom = memory.render_geometry_mask()
            context = ctx.get()
            action_idx = random_action()

            obs_next, reward, terminated, truncated, info = env.step(
                DISCRETE_ACTIONS[action_idx])
            done = terminated or truncated

            frame_next = frame_to_tensor(obs_next)
            pose_next = torch.tensor(memory.get_pose(), dtype=torch.float32)

            replay.push(Transition(
                target_frame=frame_next,
                context_frames=context,
                geometry=geom,
                action=action_idx,
                pose_current=pose_curr,
                pose_next=pose_next,
            ))
            ctx.push(frame_next)

            # --- Train ---
            if len(replay) >= MIN_BUFFER_SIZE:
                batch = replay.sample(min(BATCH_SIZE, len(replay)))
                tgt = batch['target_frame'].to(device)
                cxt = batch['context_frames'].to(device)
                geo = batch['geometry'].to(device)
                act = batch['action'].to(device)
                pc = batch['pose_current'].to(device)
                pn = batch['pose_next'].to(device)

                # Observation loss
                obs_loss = obs_net.compute_loss(tgt, cxt, geo, act)

                # UNet features for dynamics
                B = tgt.shape[0]
                mid_t = torch.full((B,), NUM_TRAIN_TIMESTEPS // 2,
                                   device=device, dtype=torch.long)
                noise_f = torch.randn_like(tgt)
                x_noisy_f = obs_net.scheduler.add_noise(tgt, noise_f, mid_t)
                unet_f = obs_net.get_mid_features(x_noisy_f, mid_t, cxt, geo, act)

                # Dynamics loss
                geo_1d = geo[:, 0, IMG_SIZE // 2, :]  # center row slice
                dyn_loss = dyn_net.compute_loss(pc, act, geo_1d, unet_f, pn)

                loss = obs_loss + DYNAMICS_LOSS_WEIGHT * dyn_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(obs_net.parameters()) + list(dyn_net.parameters()), 1.0)
                optimizer.step()

                steps += 1
                cum_obs += obs_loss.item()
                cum_dyn += dyn_loss.item()

                if steps % LOG_EVERY == 0:
                    print(f"  Step {steps} | "
                          f"Obs: {cum_obs/LOG_EVERY:.4f} | "
                          f"Dyn: {cum_dyn/LOG_EVERY:.4f} | "
                          f"Buf: {len(replay)}")
                    cum_obs = cum_dyn = 0.0

                if steps % SAVE_EVERY == 0:
                    torch.save(obs_net.state_dict(),
                               os.path.join(args.save_dir, "obs_net.pth"))
                    torch.save(dyn_net.state_dict(),
                               os.path.join(args.save_dir, "dyn_net.pth"))
                    print(f"  → Saved checkpoint at step {steps}")

                if args.max_steps and steps >= args.max_steps:
                    done = True
                    break

        if args.max_steps and steps >= args.max_steps:
            break

    # Final save
    torch.save(obs_net.state_dict(), os.path.join(args.save_dir, "obs_net.pth"))
    torch.save(dyn_net.state_dict(), os.path.join(args.save_dir, "dyn_net.pth"))
    env.close()
    print(f"\nDone. {steps} training steps.")


# ═══════════════════════════════════════════════════════════════════
# INFERENCE — Autoregressive rollout with memory-based geometry
# ═══════════════════════════════════════════════════════════════════
@torch.no_grad()
def infer(args):
    import matplotlib.pyplot as plt

    device = args.device
    env = gym.make("CarRacing-v2", render_mode="rgb_array")
    memory = CarRacingMemory(env)

    obs_net = ObservationModule().to(device)
    obs_net.load_state_dict(torch.load(
        os.path.join(args.save_dir, "obs_net.pth"), map_location=device))
    obs_net.eval()

    dyn_net = DynamicsModule(unet_feat_dim=256).to(device)
    dyn_net.load_state_dict(torch.load(
        os.path.join(args.save_dir, "dyn_net.pth"), map_location=device))
    dyn_net.eval()

    obs, info = env.reset()
    ctx = ContextTracker()
    ctx.push(frame_to_tensor(obs))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#1a1a2e')
    plt.ion()

    for step in range(args.infer_steps):
        action_idx = random_action()
        action_t = torch.tensor([action_idx], device=device)
        geom = memory.render_geometry_mask().unsqueeze(0).to(device)
        context = ctx.get().unsqueeze(0).to(device)

        # Diffusion reverse process
        latents = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
        obs_net.scheduler.set_timesteps(NUM_INFERENCE_STEPS)
        for t in obs_net.scheduler.timesteps:
            t_tensor = torch.tensor([t], device=device)
            v_pred = obs_net(latents, t_tensor, context, geom, action_t)
            latents = obs_net.scheduler.step(v_pred, t, latents).prev_sample

        gen_frame = latents.clamp(-1, 1)

        # Step real env for comparison
        obs_real, _, term, trunc, _ = env.step(DISCRETE_ACTIONS[action_idx])
        if term or trunc:
            obs_real, _ = env.reset()
            ctx.reset()

        # Visualize
        gen_img = (gen_frame[0].cpu().permute(1, 2, 0).numpy() + 1) / 2
        real_img = obs_real / 255.0
        geom_img = geom[0, 0].cpu().numpy()

        for ax in axes: ax.clear()
        axes[0].imshow(real_img); axes[0].set_title("Real", color='w')
        axes[1].imshow(np.clip(gen_img, 0, 1)); axes[1].set_title("Generated", color='w')
        axes[2].imshow(geom_img, cmap='hot'); axes[2].set_title("Geometry Mask", color='w')
        for ax in axes: ax.axis('off')
        plt.tight_layout(); plt.pause(0.05)

        ctx.push(gen_frame[0].cpu())

    plt.ioff(); plt.show()
    env.close()
    print("Inference complete!")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "infer"], default="train")
    p.add_argument("--episodes", type=int, default=EPISODES)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--infer_steps", type=int, default=100)
    p.add_argument("--save_dir", type=str, default="checkpoints_car")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        infer(args)
