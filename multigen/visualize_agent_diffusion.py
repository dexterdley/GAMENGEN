import os
import sys
import time
import itertools as it
import numpy as np
import torch
import torch.nn as nn
import skimage.color
import skimage.transform
import matplotlib.pyplot as plt

import vizdoom as vzd
from observation import ObservationModule
from train_agent_diffusion import preprocess_for_agent, extract_geometry

# Include DuelQNet structure so PyTorch can unpickle the model
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

setattr(sys.modules['__main__'], 'DuelQNet', DuelQNet)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ==========================
    # 1. Initialize ViZDoom
    # ==========================
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "simpler_basic.cfg"))
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    # Using small res to speed up the diffusion generation loop
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120) 
    game.set_depth_buffer_enabled(True)
    game.set_window_visible(False)
    game.init()

    n_buttons = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n_buttons)]
    action_names = ["NONE", "LEFT", "RIGHT", "LEFT+RIGHT", "ATTACK", "ATTACK+LEFT", "ATTACK+RIGHT", "ALL"]
    
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
    # For a proper test, you would load pre-trained weights here
    # Since we are just visually testing the pipeline, we'll use an untrained/briefly-trained model
    resolution = (120, 160) # Native screen resolution
    context_frames = 4
    observation_net = ObservationModule(context_frames=context_frames, resolution=resolution, action_dim=len(actions)).to(device)
    
    try:
        observation_net.load_state_dict(torch.load("obs_model.pth", map_location=device))
        print("Successfully loaded trained observation model weights.")
    except Exception as e:
        print(f"Could not load trained weights: {e}. Visualization will use untrained model.")
        
    observation_net.eval()
    
    context_buffer = torch.zeros(1, 3 * context_frames, resolution[0], resolution[1], device=device)

    # ==========================
    # 4. Interactive Visualization Loop
    # ==========================
    print("\n--- Starting Interactive Visualization ---")
    game.new_episode()
    plt.ion() # Interactive mode
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    inference_steps = 50
    
    for i in range(inference_steps):
        state = game.get_state()
        if game.is_episode_finished() or state is None:
            print("Episode completed, resetting environment...")
            game.new_episode()
            context_buffer.zero_()
            state = game.get_state()
            if state is None:
                break
            
        # Agent decides action based on the TRUE game state to drive exploration
        agent_input = preprocess_for_agent(state.screen_buffer)
        agent_input_tensor = torch.from_numpy(agent_input).reshape(1, 1, 30, 45).to(device)
        with torch.no_grad():
            action_idx = agent(agent_input_tensor).argmax().item()
            action_tensor = torch.tensor([action_idx], device=device)
            
        geometry = extract_geometry(state.depth_buffer, resolution).to(device)
        
        # DIFFUSION GENERATION (Generating the next frame based on context and chosen action)
        latents = torch.randn(1, 3, resolution[0], resolution[1], device=device)
        # Use 20 steps for somewhat recognizable noise structures
        observation_net.scheduler.set_timesteps(20) 
        
        start_t = time.time()
        with torch.no_grad():
            for t in observation_net.scheduler.timesteps:
                t_tensor = torch.tensor([t], device=device)
                
                # Scale model input if required by scheduler
                latent_model_input = observation_net.scheduler.scale_model_input(latents, t)
                
                v_pred = observation_net(x_noisy=latent_model_input, timesteps=t_tensor, context_frames=context_buffer, geometry_disparity=geometry, actions=action_tensor)
                
                # compute previous image: x_t -> x_t-1
                # Since we trained with v_prediction:
                latents = observation_net.scheduler.step(v_pred, t, latents).prev_sample
                
        gen_time = time.time() - start_t
        
        gen_frame = latents.clamp(-1.0, 1.0)
        
        # Format frames for matplotlib
        # generated frame is (1, 3, H, W) in [-1, 1]. Convert to (H, W, 3) in [0, 1]
        gen_img = gen_frame[0].cpu().permute(1, 2, 0).numpy()
        gen_img = (gen_img + 1.0) / 2.0
        
        true_img = state.screen_buffer
        
        # Display
        axes[0].clear()
        axes[0].imshow(true_img)
        axes[0].set_title("True ViZDoom State (Agent sees this)")
        axes[0].axis('off')
        
        axes[1].clear()
        axes[1].imshow(gen_img)
        axes[1].set_title(f"Diffusion Prediction\nAction: {action_names[action_idx]}")
        axes[1].axis('off')
        
        plt.suptitle(f"Step {i+1} | Gen Time: {gen_time:.2f}s")
        plt.pause(0.1) # Brief pause to allow UI update
        
        # Update context
        context_buffer = torch.roll(context_buffer, shifts=-3, dims=1)
        context_buffer[:, -3:, :, :] = gen_frame
        
        # Step game to keep state synced for agent
        game.set_action(actions[action_idx])
        for _ in range(12):
            game.advance_action()

    plt.ioff()
    plt.show() # Keep the final frame open
    game.close()
    print("Visualization Complete!")

if __name__ == "__main__":
    main()
