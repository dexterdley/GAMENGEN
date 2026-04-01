import torch
from memory import Map, PlayerState
from engine import MultiGenEnv
import time

def run_simulation(steps=10):
    print("=" * 50)
    print("Starting MultiGen Simple Engine Test")
    print("=" * 50)
    
    # 1. Initialize simple square map
    test_map = Map.create_simple_box(size=20.0)
    
    # 2. Place player in center looking along X axis
    initial_pose = PlayerState(x=10.0, y=10.0, yaw=0.0)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running simulation on device: {device}")
    
    # 3. Initialize Environment
    # Lower resolution for fast CPU testing
    env = MultiGenEnv(
        game_map=test_map,
        initial_pose=initial_pose,
        context_frames=4,
        resolution=32,
        device=device
    )
    
    obs_dict = env.reset()
    
    print(f"\nInitial State:")
    print(f"Pose: ({obs_dict['pose'][0]:.2f}, {obs_dict['pose'][1]:.2f}, {obs_dict['pose'][2]:.2f})")
    print(f"Geometry tensor shape: {obs_dict['geometry_disparity'].shape}")
    print(f"Visual context shape: {obs_dict['visual_context'].shape}")
    
    # 4. Run loop
    for step_idx in range(steps):
        print(f"\n--- Step {step_idx + 1}/{steps} ---")
        
        # Simulate a random discrete action (e.g. 0-15)
        action = torch.randint(0, 16, (1,)).item()
        print(f"Agent Action: {action}")
        
        start_time = time.time()
        
        # Step environment
        frame, obs_dict = env.step(action_id=action)
        
        end_time = time.time()
        
        print(f"Generated frame: min={frame.min().item():.3f}, max={frame.max().item():.3f}")
        print(f"Updated Pose: ({obs_dict['pose'][0]:.2f}, {obs_dict['pose'][1]:.2f}, {obs_dict['pose'][2]:.2f})")
        print(f"Step latency: {(end_time - start_time)*1000:.1f} ms")

    print("\n" + "=" * 50)
    print("Simulation Test Completed Successfully!")
    print("=" * 50)

if __name__ == "__main__":
    run_simulation(10)
