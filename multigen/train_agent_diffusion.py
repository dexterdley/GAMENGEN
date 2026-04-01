import os
import sys
import time
import itertools as it
import numpy as np
import torch
import torch.nn as nn
import skimage.color
import skimage.transform

import vizdoom as vzd
from observation import ObservationModule
import math

# 1. Provide the neural network architecture from test_pytorch.py so we can unpickle model-doom.pth
class DuelQNet(nn.Module):
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

# Hack to allow torch.load to find DuelQNet in the root namespace
setattr(sys.modules['__main__'], 'DuelQNet', DuelQNet)

def preprocess_for_agent(img_rgb):
    """ Converts ViZDoom RGB (H, W, C) buffer to grayscale 30x45 for the RL agent """
    # ViZDoom RGB24 buffer is already (H, W, C)
    img_gray = skimage.color.rgb2gray(img_rgb)
    img_resized = skimage.transform.resize(img_gray, (30, 45))
    return img_resized.astype(np.float32)

def extract_geometry(depth_buffer, resolution):
    """Extracts depth buffer to 1D disparity and expands to spatial tensor."""
    if depth_buffer is None:
        if isinstance(resolution, int):
            return torch.zeros(1, 1, resolution, resolution)
        return torch.zeros(1, 1, resolution[0], resolution[1])
        
    mid_y = depth_buffer.shape[0] // 2
    scanline_1d = depth_buffer[mid_y, :]
    scanline_1d = np.array(scanline_1d, dtype=np.float32)
    disparity = scanline_1d / 255.0 # Max normalized
    
    # Resize to UNet resolution width
    target_w = resolution if isinstance(resolution, int) else resolution[1]
    target_h = resolution if isinstance(resolution, int) else resolution[0]
    
    x_old = np.linspace(0, 1, len(disparity))
    x_new = np.linspace(0, 1, target_w)
    disparity_resized = np.interp(x_new, x_old, disparity)
    
    # Broadcast to spatial tensor (1, 1, H, W)
    disp_tensor = torch.tensor(disparity_resized, dtype=torch.float32)
    return disp_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 1, target_h, 1)

def extract_frame_for_diffusion(rgb_buffer, resolution=None):
    """ Converts rgb_buffer into diffusion tensor format, normalizes to [-1, 1] """
    # Convert uint8 RGB [0, 255] to float32 [-1, 1] without resizing
    img_float = rgb_buffer.astype(np.float32) / 255.0
    img_normalized = (img_float * 2.0) - 1.0
    t = torch.tensor(img_normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return t

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ==========================
    # 1. Initialize ViZDoom
    # ==========================
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "simpler_basic.cfg"))
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    # Using small res to speed up training processing
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120) 
    game.set_depth_buffer_enabled(True)
    game.set_window_visible(False)
    game.init()

    # simpler_basic has 3 buttons -> 8 possible actions
    n_buttons = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n_buttons)]
    
    # ==========================
    # 2. Load RL Agent 
    # ==========================
    agent_path = "../model-doom.pth"
    print(f"Loading agent from {agent_path}")
    agent = torch.load(agent_path, map_location=device, weights_only=False)
    agent.eval()

    # ==========================
    # 3. Initialize Observation Module
    # ==========================
    resolution = (120, 160) # Native screen resolution
    context_frames = 4
    observation_net = ObservationModule(context_frames=context_frames, resolution=resolution, action_dim=len(actions)).to(device)
    optimizer = torch.optim.Adam(observation_net.parameters(), lr=3e-4) # Lower learning rate

    # ==========================
    # 4. Data Collection & Training Loop
    # ==========================
    print("\n--- Starting Agent-driven Diffusion Training ---")
    episodes = 200 # Lower for faster test 
    batch_size = 16
    
    # Buffers to store transitions
    context_buffer = torch.zeros(1, 3 * context_frames, resolution[0], resolution[1], device=device)
    
    training_steps = 0
    accumulated_loss = 0.0
    observation_net.train()
    optimizer.zero_grad()
    
    for ep in range(episodes):
        game.new_episode()
        # Reset context
        context_buffer.zero_()
        
        while not game.is_episode_finished():
            state = game.get_state()
            if state is None:
                break
                
            # --- AGENT ACTION ---
            agent_input = preprocess_for_agent(state.screen_buffer)
            agent_input_tensor = torch.from_numpy(agent_input).reshape(1, 1, 30, 45).to(device)
            with torch.no_grad():
                best_action_index = agent(agent_input_tensor).argmax().item()
            
            # --- GET CURRENT GEOMETRY & FRAME ---
            geometry = extract_geometry(state.depth_buffer, resolution).to(device)
            frame_n = extract_frame_for_diffusion(state.screen_buffer, resolution).to(device)
            
            # Form action tensor
            action_tensor = torch.tensor([best_action_index], device=device)
            
            # --- STEP ENGINE ---
            game.set_action(actions[best_action_index])
            for _ in range(12): # frame_repeat from test_pytorch.py
                game.advance_action()
                
            next_state = game.get_state()
            if next_state is None:
                break
                
            # Next frame is our target
            target_frame = extract_frame_for_diffusion(next_state.screen_buffer, resolution).to(device)
            
            # --- TRAIN STEP ---
            # compute_loss(target_frame, context_frames, geometry_disparity, actions)
            loss = observation_net.compute_loss(target_frame, context_buffer, geometry, action_tensor)
            
            # Loss division for gradient accumulation
            (loss / batch_size).backward()
            accumulated_loss += loss.item()
            
            training_steps += 1
            
            if training_steps % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
                
            if training_steps % 10 == 0:
                avg_loss = accumulated_loss / 10.0
                print(f"Ep {ep+1} | Step {training_steps} | Avg Diffusion Loss: {avg_loss:.4f} | Last Action: {best_action_index}")
                accumulated_loss = 0.0
            
            # --- UPDATE CONTEXT ---
            # target frame has 3 channels. Append to the end, drop the front 3.
            context_buffer = torch.roll(context_buffer, shifts=-3, dims=1)
            context_buffer[:, -3:, :, :] = target_frame.clone().detach()



    print(f"Training completed for {training_steps} steps.")
    
    # Save the trained model weights so the visualizer can use them
    torch.save(observation_net.state_dict(), "obs_model.pth")
    print("Saved trained checkpoint to obs_model.pth")
    
    # ==========================
    # 5. Testing Generative Model
    # ==========================
    print("\n--- Starting Generative Inference Test ---")
    observation_net.eval()
    game.new_episode()
    context_buffer.zero_()
    
    # Let's run autoregressive generation for 10 steps based on agent commands
    for i in range(10):
        state = game.get_state()
        if game.is_episode_finished() or state is None:
            game.new_episode()
            context_buffer.zero_()
            state = game.get_state()
            if state is None:
                break
            
        agent_input = preprocess_for_agent(state.screen_buffer)
        agent_input_tensor = torch.from_numpy(agent_input).reshape(1, 1, 30, 45).to(device)
        with torch.no_grad():
            action_idx = agent(agent_input_tensor).argmax().item()
            action_tensor = torch.tensor([action_idx], device=device)
            
        geometry = extract_geometry(state.depth_buffer, resolution).to(device)
        
        # DIFFUSION GENERATION
        latents = torch.randn(1, 3, resolution[0], resolution[1], device=device)
        observation_net.scheduler.set_timesteps(20) # More timesteps for better inference quality
        
        with torch.no_grad():
            for t in observation_net.scheduler.timesteps:
                t_tensor = torch.tensor([t], device=device)
                
                # Scale model input if required by scheduler
                latent_model_input = observation_net.scheduler.scale_model_input(latents, t)
                
                v_pred = observation_net(x_noisy=latent_model_input, timesteps=t_tensor, context_frames=context_buffer, geometry_disparity=geometry, actions=action_tensor)
                
                # compute previous image: x_t -> x_t-1
                # Since we trained with v_prediction:
                latents = observation_net.scheduler.step(v_pred, t, latents).prev_sample
                
        gen_frame = latents.clamp(-1.0, 1.0)
        print(f"Inference Step {i+1} | Generated Frame diff min/max: {gen_frame.min().item():.3f}, {gen_frame.max().item():.3f} | Agent Action: {action_idx}")
        
        # Update context
        context_buffer = torch.roll(context_buffer, shifts=-3, dims=1)
        context_buffer[:, -3:, :, :] = gen_frame
        
        # Step game to keep state synced for agent
        game.set_action(actions[action_idx])
        for _ in range(12):
            game.advance_action()

    game.close()
    print("Agent-Driven Diffusion Test Successful!")

if __name__ == "__main__":
    main()
