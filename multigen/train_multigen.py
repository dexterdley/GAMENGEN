"""
train_multigen.py — Joint Observation + Dynamics training for MultiGen.

Bridges the existing pixel-space diffusion training (train_agent_diffusion.py)
with MultiGen's Memory–Observation–Dynamics architecture.

Usage:
    python train_multigen.py [--episodes 200] [--max_steps 5000] [--device cuda]
"""

import os
import sys
import math
import argparse
import itertools as it
import time
import numpy as np
import torch
import torch.nn as nn
import skimage.color
import skimage.transform

import vizdoom as vzd

from tqdm import tqdm
current_dir = os.path.dirname(os.path.abspath(__file__))
ppo_dir = os.path.join(current_dir, "..", "ViZDoomPPO")
sys.path.append(ppo_dir)
# --------------------------------------------------------------

from stable_baselines3 import PPO
from common import envs # Now Python will find this!

from observation import ObservationModule
from dynamics import DynamicsModule
from dataset_multigen import (
    ReplayBuffer, Transition, ContextTracker,
    extract_frame_for_diffusion, extract_geometry, extract_pose,
)
from config_multigen import (
    RESOLUTION, CONTEXT_FRAMES, ACTION_DIM,
    CONTEXT_NOISE_SCALE, DYNAMICS_LOSS_WEIGHT,
    UNET_MID_FEATURES_DIM,
    LEARNING_RATE, BATCH_SIZE, EPISODES, FRAME_REPEAT,
    REPLAY_BUFFER_SIZE, MIN_BUFFER_SIZE,
    SAVE_EVERY_STEPS, LOG_EVERY_STEPS,
    NUM_TRAIN_TIMESTEPS, PREDICTION_TYPE,
)


# ---------------------------------------------------------------------------
# RL Agent (for driving exploration)
# ---------------------------------------------------------------------------
class DuelQNet(nn.Module):
    """Dueling DQN agent architecture — must match the saved model-doom.pth."""
    def __init__(self, available_actions_count):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 8, kernel_size=3, stride=2, bias=False), nn.BatchNorm2d(8), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False), nn.BatchNorm2d(8), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False), nn.BatchNorm2d(8), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        self.state_fc = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage_fc = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, available_actions_count))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 192)
        x1 = x[:, :96]
        x2 = x[:, 96:]
        state_value = self.state_fc(x1).reshape(-1, 1)
        advantage_values = self.advantage_fc(x2)
        x = state_value + (advantage_values - advantage_values.mean(dim=1).reshape(-1, 1))
        return x

# Allow torch.load to unpickle DuelQNet
setattr(sys.modules['__main__'], 'DuelQNet', DuelQNet)


def preprocess_for_agent(img_rgb):
    """Convert ViZDoom RGB buffer to grayscale 30x45 for the RL agent."""
    img_gray = skimage.color.rgb2gray(img_rgb)
    img_resized = skimage.transform.resize(img_gray, (30, 45))
    return img_resized.astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="MultiGen joint training")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max training gradient steps. None = run all episodes.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--dynamics_weight", type=float, default=DYNAMICS_LOSS_WEIGHT)
    parser.add_argument("--context_noise", type=float, default=CONTEXT_NOISE_SCALE)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints_multigen")
    parser.add_argument("--agent_path", type=str, default="../ViZDoomPPO/logs/models/deathmatch_simple/best_model.zip")
    parser.add_argument("--scenario", type=str, default=None,
                        help="ViZDoom scenario config path. Default: simpler_basic.cfg")
    return parser.parse_args()


def init_vizdoom(scenario_path=None):
    """Initialize ViZDoom with depth buffer and game variables for pose extraction."""
    game = vzd.DoomGame()
    if scenario_path:
        game.load_config(scenario_path)
    else:
        # Default to deathmatch_simple which matches the PPO agent's training config
        ppo_scenario = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "ViZDoomPPO", "scenarios", "deathmatch_simple.cfg"
        )
        if os.path.exists(ppo_scenario):
            game.load_config(ppo_scenario)
        else:
            game.load_config(os.path.join(vzd.scenarios_path, "simpler_basic.cfg"))

    # Deathmatch args (same as DoomWithBots in PPO training)
    game.add_game_args(
        "-host 1 -deathmatch +viz_nocheat 0 +cl_run 1 +name AGENT "
        "+colorset 0 +sv_forcerespawn 1 +sv_respawnprotect 1 "
        "+sv_nocrouch 1 +sv_noexit 1"
    )

    game.set_screen_format(vzd.ScreenFormat.RGB24)
    # Use 320x240 to match PPO agent's training resolution
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.set_depth_buffer_enabled(True)

    # Game variables for pose extraction
    game.add_available_game_variable(vzd.GameVariable.POSITION_X)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
    game.add_available_game_variable(vzd.GameVariable.ANGLE)

    game.set_window_visible(False)
    game.init()

    # Add bots for deathmatch
    n_bots = 4
    game.send_game_command('removebots')
    for _ in range(n_bots):
        game.send_game_command('addbot')

    return game


def compute_dynamics_loss(dynamics_module, pose_current, action, geometry_1d,
                          unet_features, pose_next):
    """
    Compute dynamics loss: MSE on translation + wrapped angle error on yaw (§3.3.3).
    """
    delta_pred = dynamics_module(pose_current, action, geometry_1d, unet_features)

    # GT delta
    dx = pose_next[:, 0] - pose_current[:, 0]
    dy = pose_next[:, 1] - pose_current[:, 1]
    dyaw = (pose_next[:, 2] - pose_current[:, 2] + math.pi) % (2 * math.pi) - math.pi
    delta_gt = torch.stack([dx, dy, dyaw], dim=1)

    loss = nn.functional.mse_loss(delta_pred, delta_gt)
    return loss, delta_pred


def main():
    args = parse_args()
    device = args.device
    print(f"Device: {device}")
    print(f"Config: episodes={args.episodes}, batch_size={args.batch_size}, "
          f"lr={args.lr}, dynamics_weight={args.dynamics_weight}, "
          f"context_noise={args.context_noise}")

    # ========================
    # 1. Initialize ViZDoom
    # ========================
    game = init_vizdoom(args.scenario)
    n_buttons = game.get_available_buttons_size()
    
    # Build the same action space the PPO agent was trained with
    # get_available_actions filters out illegal combos (mutually exclusive buttons, etc.)
    from common.utils import get_available_actions
    button_array = np.array(game.get_available_buttons())
    actions = get_available_actions(button_array)
    n_actions = len(actions)
    print(f"ViZDoom initialized: {n_buttons} buttons → {n_actions} legal action combos")

    # ========================
    # 2. Load RL Agent
    # ========================
    print(f"Loading RL agent from {args.agent_path}")
    agent = PPO.load(args.agent_path, device=device)
    agent.policy.eval()

    # ========================
    # 3. Initialize Models
    # ========================
    observation_net = ObservationModule(
        context_frames=CONTEXT_FRAMES,
        resolution=RESOLUTION,
        action_dim=n_actions,
    ).to(device)
    observation_net.train()

    dynamics_net = DynamicsModule(
        action_dim=n_actions,
        geometry_dim=RESOLUTION[1],  # 1D disparity width
        unet_feature_dim=UNET_MID_FEATURES_DIM,
    ).to(device)
    dynamics_net.train()

    # Joint optimizer over both modules
    optimizer = torch.optim.Adam(
        list(observation_net.parameters()) + list(dynamics_net.parameters()),
        lr=args.lr,
    )

    # ========================
    # 4. Replay Buffer + Context
    # ========================
    replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE)
    context_tracker = ContextTracker(CONTEXT_FRAMES, RESOLUTION)

    # ========================
    # 5. Training Loop
    # ========================
    print("\n--- Starting MultiGen Joint Training ---")
    os.makedirs(args.save_dir, exist_ok=True)

    training_steps = 0
    total_obs_loss = 0.0
    total_dyn_loss = 0.0
    collection_steps = 0

    for ep in tqdm(range(args.episodes), desc="Episodes"):
        game.new_episode()
        context_tracker.reset()

        # We need the first state to get pose_current
        state = game.get_state()
        if state is None:
            continue

        prev_pose = extract_pose(state)
        prev_frame = extract_frame_for_diffusion(state.screen_buffer)
        context_tracker.push(prev_frame)

        while not game.is_episode_finished():
            # Handle player death in deathmatch
            if game.is_player_dead():
                game.respawn_player()
                context_tracker.reset()

            state = game.get_state()
            if state is None:
                break

            # --- Agent selects action ---
            #agent_input = preprocess_for_agent(state.screen_buffer)
            #agent_input_tensor = torch.from_numpy(agent_input).reshape(1, 1, 30, 45).to(device)
            #with torch.no_grad():
            #    action_idx = agent(agent_input_tensor).argmax().item()
            
            obs = envs.default_frame_processor(state.screen_buffer)
            # PPO was trained with VecTransposeImage which transposes (H,W,C) → (C,H,W)
            obs = np.transpose(obs, (2, 0, 1))
            
            # predict() returns the action and the RNN states (which we ignore with _)
            action_idx, _ = agent.predict(obs, deterministic=True)
            action_idx = int(action_idx)

            # --- Capture current state ---
            current_pose = extract_pose(state)
            current_geometry = extract_geometry(state.depth_buffer, RESOLUTION)
            current_context = context_tracker.get()

            # --- Step ViZDoom ---
            reward = game.make_action(actions[action_idx], FRAME_REPEAT)

            if game.is_episode_finished():
                break
            
            next_state = game.get_state()
            if next_state is None:
                break

            # --- Capture next state ---
            next_frame = extract_frame_for_diffusion(next_state.screen_buffer)
            next_pose = extract_pose(next_state)

            # --- Store transition ---
            transition = Transition(
                target_frame=next_frame,
                context_frames=current_context,
                geometry=current_geometry,                # (1, H, W)
                action=action_idx,
                pose_current=current_pose,
                pose_next=next_pose,
            )
            replay_buffer.push(transition)
            collection_steps += 1

            # --- Update context ---
            context_tracker.push(next_frame)

            # --- Train if enough data ---
            if len(replay_buffer) >= MIN_BUFFER_SIZE:
                batch = replay_buffer.sample(min(args.batch_size, len(replay_buffer)))

                # Move to device
                target = batch['target_frame'].to(device)
                context = batch['context_frames'].to(device)
                geometry = batch['geometry'].to(device)
                action_batch = batch['action'].to(device)
                pose_curr = batch['pose_current'].to(device)
                pose_next_batch = batch['pose_next'].to(device)

                # --- Observation loss (with noised-context augmentation) ---
                obs_loss, v_pred = observation_net.compute_loss_with_noised_context(
                    target_frame=target,
                    context_frames=context,
                    geometry_disparity=geometry,
                    actions=action_batch,
                    max_noise_scale=args.context_noise,
                )

                # --- Extract UNet mid-block features for dynamics (detached) ---
                # We need to run a forward pass at a specific timestep to get features.
                # Use the midpoint timestep for stable features.
                B = target.shape[0]
                mid_t = torch.tensor(
                    [observation_net.scheduler.config.num_train_timesteps // 2] * B,
                    device=device,
                )
                noise_for_feat = torch.randn_like(target)
                x_noisy_for_feat = observation_net.scheduler.add_noise(target, noise_for_feat, mid_t)

                unet_features = observation_net.get_intermediate_features(
                    x_noisy_for_feat, mid_t, context, geometry, action_batch
                )  # (B, 512), already detached

                # --- Dynamics loss ---
                # 1D geometry for dynamics: take center row of the spatial disparity
                geometry_1d = geometry[:, 0, 0, :]  # (B, W)

                dyn_loss, delta_pred = compute_dynamics_loss(
                    dynamics_net,
                    pose_curr, action_batch, geometry_1d,
                    unet_features, pose_next_batch,
                )

                # --- Joint loss ---
                total_loss = obs_loss + args.dynamics_weight * dyn_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(observation_net.parameters()) + list(dynamics_net.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()

                training_steps += 1
                total_obs_loss += obs_loss.item()
                total_dyn_loss += dyn_loss.item()

                if training_steps % LOG_EVERY_STEPS == 0:
                    avg_obs = total_obs_loss / LOG_EVERY_STEPS
                    avg_dyn = total_dyn_loss / LOG_EVERY_STEPS
                    print(
                        f"Ep {ep+1}/{args.episodes} | "
                        f"Step {training_steps} | "
                        f"Obs Loss: {avg_obs:.4f} | "
                        f"Dyn Loss: {avg_dyn:.4f} | "
                        f"Buffer: {len(replay_buffer)} | "
                        f"Collected: {collection_steps}"
                    )
                    total_obs_loss = 0.0
                    total_dyn_loss = 0.0

                if training_steps % SAVE_EVERY_STEPS == 0:
                    ckpt_path = os.path.join(args.save_dir, f"step_{training_steps}")
                    os.makedirs(ckpt_path, exist_ok=True)
                    torch.save(observation_net.state_dict(),
                               os.path.join(ckpt_path, "observation.pth"))
                    torch.save(dynamics_net.state_dict(),
                               os.path.join(ckpt_path, "dynamics.pth"))
                    print(f"  → Saved checkpoint to {ckpt_path}")

                if args.max_steps and training_steps >= args.max_steps:
                    break

        if args.max_steps and training_steps >= args.max_steps:
            print(f"Reached max_steps={args.max_steps}, stopping.")
            break

    # ========================
    # 6. Save Final Checkpoint
    # ========================
    final_path = os.path.join(args.save_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    torch.save(observation_net.state_dict(), os.path.join(final_path, "observation.pth"))
    torch.save(dynamics_net.state_dict(), os.path.join(final_path, "dynamics.pth"))
    print(f"\nTraining complete. {training_steps} steps. Saved to {final_path}")

    game.close()


if __name__ == "__main__":
    main()
