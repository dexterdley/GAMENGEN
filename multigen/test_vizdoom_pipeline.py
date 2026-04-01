import torch
import time
from vizdoom_engine import VizDoomMultiGenEnv

def run_vizdoom_pipeline(steps=15):
    print("=" * 60)
    print("Starting ViZDoom + MultiGen Integration Test")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    
    env = VizDoomMultiGenEnv(context_frames=4, resolution=32, device=device)
    
    obs_dict = env.reset()
    if obs_dict is None:
        print("Failed to initialize game.")
        return
        
    print("\nInitial True State:")
    print(f"Pose (X, Y, Yaw): ({obs_dict['pose'][0]:.1f}, {obs_dict['pose'][1]:.1f}, {obs_dict['pose'][2]:.2f})")
    print(f"Depth Tensor shape: {obs_dict['geometry_disparity'].shape}")
    
    try:
        for step_idx in range(steps):
            print(f"\n--- Step {step_idx + 1}/{steps} ---")
            
            # Action logic: Turn left (0), Turn right (1), Move Forward (2)
            # Favor moving forward to see visual changes
            if skip_prob := torch.rand(1).item() > 0.7:
                action = torch.randint(0, 2, (1,)).item() # Turn
            else:
                action = 2 # Forward
                
            action_name = ["TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD"][action]
            print(f"Action: {action_name} ({action})")
            
            start_time = time.time()
            frame, new_obs = env.step(action_id=action)
            end_time = time.time()
            
            if new_obs is None:
                print("Episode finished early.")
                break
                
            print(f"Latent Frame generated: min={frame.min().item():.3f}, max={frame.max().item():.3f}, shape={frame.shape}")
            
            # Show True vs Dynamics Model predicted pose
            t_pose = new_obs['pose']
            p_pose = new_obs['predicted_pose']
            print(f"True Pose:      ({t_pose[0]:.1f}, {t_pose[1]:.1f}, {t_pose[2]:.2f})")
            print(f"Predicted Pose: ({p_pose[0]:.1f}, {p_pose[1]:.1f}, {p_pose[2]:.2f})")
            print(f"Step latency:   {(end_time - start_time)*1000:.1f} ms")
            
    finally:
        env.close()
        
    print("\n" + "=" * 60)
    print("ViZDoom Integration Test Completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_vizdoom_pipeline(10)
