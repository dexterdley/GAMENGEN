"""
infer_multigen.py — Autoregressive inference with level design.

Runs the MultiGen inference loop (Section 3.4):
  1. Query MemoryModule for geometric readout (ray-traced disparity from map + pose)
  2. Generate next frame via ObservationModule (diffusion reverse process)
  3. Update pose via DynamicsModule
  4. Shift context window

Supports two geometry modes:
  - MAP mode: uses MemoryModule ray-tracing from a user-defined Map
  - VIZDOOM mode: uses live ViZDoom depth buffer (for comparison/validation)

Usage:
    python infer_multigen.py --checkpoint checkpoints_multigen/final --steps 50
    python infer_multigen.py --checkpoint checkpoints_multigen/final --steps 50 --use_vizdoom
"""

import os
import sys
import argparse
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from observation import ObservationModule
from dynamics import DynamicsModule
from memory import Map, PlayerState, MemoryModule
from config_multigen import (
    RESOLUTION, CONTEXT_FRAMES, ACTION_DIM,
    UNET_MID_FEATURES_DIM, NUM_INFERENCE_STEPS,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MultiGen inference with level design")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint dir")
    parser.add_argument("--steps", type=int, default=50, help="Number of autoregressive steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_vizdoom", action="store_true",
                        help="Use live ViZDoom depth instead of map ray-tracing")
    parser.add_argument("--action_sequence", type=str, default=None,
                        help="Comma-separated action indices, e.g. '2,2,2,0,2,2,1'")
    parser.add_argument("--save_frames", type=str, default=None,
                        help="Directory to save generated frames")
    return parser.parse_args()


def create_default_map():
    """Creates a simple L-shaped corridor for testing level design."""
    # L-shaped room:
    #  (0,10)----(5,10)
    #     |         |
    #     |  (5,5)--+---(15,5)
    #     |               |
    #  (0,0)----------(15,0)
    v = [
        (0.0, 0.0),    # 0
        (15.0, 0.0),   # 1
        (15.0, 5.0),   # 2
        (5.0, 5.0),    # 3
        (5.0, 10.0),   # 4
        (0.0, 10.0),   # 5
    ]
    e = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    return Map(vertices=v, edges=e)


def draw_minimap(ax, game_map, player_state, title="Minimap"):
    """Draw a top-down minimap with player position and heading."""
    ax.clear()
    lines = game_map.get_lines()
    for (x1, y1), (x2, y2) in lines:
        ax.plot([x1, x2], [y1, y2], 'w-', linewidth=2)

    # Player position
    px, py, yaw = player_state.get_pose()
    ax.plot(px, py, 'ro', markersize=8)

    # Heading arrow
    arrow_len = 1.5
    ax.arrow(px, py, arrow_len * math.cos(yaw), arrow_len * math.sin(yaw),
             head_width=0.3, head_length=0.2, fc='red', ec='red')

    ax.set_xlim(-1, 16)
    ax.set_ylim(-1, 11)
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a2e')
    ax.set_title(title, color='white')
    ax.tick_params(colors='white')


@torch.no_grad()
def run_inference(args):
    device = args.device

    # ========================
    # 1. Load Models
    # ========================
    print(f"Loading checkpoint from {args.checkpoint}")

    # Determine action_dim from checkpoint (or use default)
    obs_state = torch.load(os.path.join(args.checkpoint, "observation.pth"),
                           map_location=device)
    # Infer action_dim from the action_embedding weight shape
    action_embed_key = "action_embedding.weight"
    if action_embed_key in obs_state:
        n_actions = obs_state[action_embed_key].shape[0]
    else:
        n_actions = ACTION_DIM

    observation_net = ObservationModule(
        context_frames=CONTEXT_FRAMES,
        resolution=RESOLUTION,
        action_dim=n_actions,
    ).to(device)
    observation_net.load_state_dict(obs_state)
    observation_net.eval()

    dynamics_net = DynamicsModule(
        action_dim=n_actions,
        geometry_dim=RESOLUTION[1],
        unet_feature_dim=UNET_MID_FEATURES_DIM,
    ).to(device)
    dynamics_net.load_state_dict(
        torch.load(os.path.join(args.checkpoint, "dynamics.pth"), map_location=device)
    )
    dynamics_net.eval()

    # ========================
    # 2. Initialize State
    # ========================
    if args.use_vizdoom:
        import vizdoom as vzd
        import itertools as it
        game = vzd.DoomGame()
        game.load_config(os.path.join(vzd.scenarios_path, "simpler_basic.cfg"))
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        game.set_depth_buffer_enabled(True)
        game.add_available_game_variable(vzd.GameVariable.POSITION_X)
        game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
        game.add_available_game_variable(vzd.GameVariable.ANGLE)
        game.set_window_visible(False)
        game.init()
        game.new_episode()
        n_buttons = game.get_available_buttons_size()
        actions_list = [list(a) for a in it.product([0, 1], repeat=n_buttons)]
        game_map = None
        memory_module = None
    else:
        game = None
        game_map = create_default_map()
        player = PlayerState(x=7.0, y=2.5, yaw=0.0)  # Start in corridor
        memory_module = MemoryModule(game_map, player, fov_deg=90,
                                     num_rays=RESOLUTION[1])

    # Context history
    context = torch.zeros(1, 3 * CONTEXT_FRAMES, RESOLUTION[0], RESOLUTION[1],
                          device=device)

    # Parse action sequence
    if args.action_sequence:
        action_list = [int(a) for a in args.action_sequence.split(',')]
        # Repeat to fill steps
        while len(action_list) < args.steps:
            action_list = action_list + action_list
        action_list = action_list[:args.steps]
    else:
        # Default: mostly forward with occasional turns
        action_list = [2] * args.steps  # action 2 = forward in simpler_basic

    # ========================
    # 3. Inference Loop (§3.4)
    # ========================
    print(f"\n--- MultiGen Inference ({args.steps} steps) ---")

    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)

    # Setup visualization
    fig, axes = plt.subplots(1, 2 if game_map else 1, figsize=(12, 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    plt.ion()
    fig.patch.set_facecolor('#0f0f23')

    for step in range(args.steps):
        action_id = action_list[step]
        action_tensor = torch.tensor([action_id], device=device)

        # 1. Get geometric readout
        if args.use_vizdoom and game:
            from dataset_multigen import extract_geometry as _extract_geom
            state = game.get_state()
            if state is None or game.is_episode_finished():
                game.new_episode()
                state = game.get_state()
            geometry = _extract_geom(state.depth_buffer, RESOLUTION).unsqueeze(0).to(device)
        else:
            geometry = memory_module.get_geometric_readout(RESOLUTION).to(device)

        # 2. Diffusion reverse process → generate next frame
        latents = torch.randn(1, 3, RESOLUTION[0], RESOLUTION[1], device=device)
        observation_net.scheduler.set_timesteps(NUM_INFERENCE_STEPS)

        f_t = torch.zeros(1, UNET_MID_FEATURES_DIM, device=device)

        for i, t in enumerate(observation_net.scheduler.timesteps):
            t_tensor = torch.tensor([t], device=device)
            v_pred = observation_net(
                x_noisy=latents, timesteps=t_tensor,
                context_frames=context, geometry_disparity=geometry,
                actions=action_tensor,
            )
            latents = observation_net.scheduler.step(v_pred, t, latents).prev_sample

            # Extract mid-block features at midpoint for dynamics
            if i == len(observation_net.scheduler.timesteps) // 2:
                f_t = observation_net.get_intermediate_features(
                    latents, t_tensor, context, geometry, action_tensor
                )

        next_frame = latents.clamp(-1.0, 1.0)

        # 3. Update pose via dynamics
        if memory_module:
            current_pose = torch.tensor(
                [list(memory_module.player.get_pose())],
                dtype=torch.float32, device=device,
            )
        else:
            current_pose = torch.zeros(1, 3, device=device)

        geometry_1d = geometry[0, 0, 0, :]  # (W,)
        delta_pose = dynamics_net(
            pose=current_pose, action=action_tensor,
            geometry=geometry_1d.unsqueeze(0), unet_features=f_t,
        )
        updated_pose = dynamics_net.apply_update(current_pose, delta_pose)

        # Apply to memory
        if memory_module:
            memory_module.player.x = updated_pose[0, 0].item()
            memory_module.player.y = updated_pose[0, 1].item()
            memory_module.player.yaw = updated_pose[0, 2].item()

        # 4. Shift context window
        context = torch.roll(context, shifts=-3, dims=1)
        context[:, -3:, :, :] = next_frame

        # Step ViZDoom if in comparison mode
        if game:
            import itertools as it
            n_buttons = game.get_available_buttons_size()
            actions_list_local = [list(a) for a in it.product([0, 1], repeat=n_buttons)]
            if action_id < len(actions_list_local):
                game.make_action(actions_list_local[action_id], 4)

        # --- Visualization ---
        gen_img = next_frame[0].cpu().permute(1, 2, 0).numpy()
        gen_img = (gen_img + 1.0) / 2.0
        gen_img = np.clip(gen_img, 0, 1)

        axes[0].clear()
        axes[0].imshow(gen_img)
        axes[0].set_title(f"Generated Frame (Step {step+1})", color='white')
        axes[0].axis('off')

        if game_map and len(axes) > 1:
            draw_minimap(axes[1], game_map, memory_module.player,
                         title=f"Map (Action: {action_id})")

        plt.tight_layout()
        plt.pause(0.05)

        # Save frame
        if args.save_frames:
            plt.savefig(os.path.join(args.save_frames, f"frame_{step:04d}.png"),
                        dpi=100, facecolor=fig.get_facecolor())

        if step % 10 == 0:
            if memory_module:
                p = memory_module.player.get_pose()
                print(f"Step {step+1}: Pose=({p[0]:.1f}, {p[1]:.1f}, {math.degrees(p[2]):.0f}°)")
            else:
                print(f"Step {step+1}: frame range [{gen_img.min():.2f}, {gen_img.max():.2f}]")

    plt.ioff()
    plt.show()

    if game:
        game.close()

    print("Inference complete!")


if __name__ == "__main__":
    run_inference(parse_args())
