# MultiGen Configuration
# Single player, single map, pixel-space diffusion

# Resolution (H, W) — matches ViZDoom native
RESOLUTION = (120, 160)

# Number of past context frames (L)
CONTEXT_FRAMES = 4

# Number of discrete actions (simpler_basic: 3 buttons → 8 combos)
ACTION_DIM = 8

# Diffusion
NUM_TRAIN_TIMESTEPS = 1000
NUM_INFERENCE_STEPS = 20
PREDICTION_TYPE = "v_prediction"

# Noised-context training (Section 3.2.4)
# Max noise scale applied to context frames during training
CONTEXT_NOISE_SCALE = 0.1

# Dynamics module (Section 3.3)
DYNAMICS_LOSS_WEIGHT = 0.1  # λ for joint obs + dynamics loss
DYNAMICS_D_MODEL = 128
DYNAMICS_NHEAD = 4
DYNAMICS_NUM_LAYERS = 2
DYNAMICS_POSE_DIM = 3  # (x, y, yaw)

# UNet mid-block feature dim (after adaptive avg pool → 1D vector)
# Matches ObservationModule's block_out_channels[-1] = 512
UNET_MID_FEATURES_DIM = 512

# Training
LEARNING_RATE = 3e-4
BATCH_SIZE = 16  # gradient accumulation batch size
EPISODES = 200
FRAME_REPEAT = 12  # ViZDoom frame skip

# Replay buffer
REPLAY_BUFFER_SIZE = 10000
MIN_BUFFER_SIZE = 256  # minimum transitions before training starts

# Checkpointing
SAVE_EVERY_STEPS = 500
LOG_EVERY_STEPS = 10
