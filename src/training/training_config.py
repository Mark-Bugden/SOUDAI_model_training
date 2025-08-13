# Training-specific defaults

# Logging
USE_LOGGING = True
LOG_LEVEL = "WARNING"  # "DEBUG", "INFO", "WARNING"

# Data loader
BATCH_SIZE = 512
NUM_WORKERS = 0
PIN_MEMORY = False
FORCE_REBUILD = False

# Training
EPOCHS = 200
LR = 1e-5
WEIGHT_DECAY = 1e-4
DEVICE = "cuda"  # "cpu", "cuda", or "auto"
USE_HUBER = False

# Model
EMBED_DIM_SINGLE = 32
EMBED_DIM_MULTI = 32
HIDDEN = (128, 64)
DROPOUT = 0.1

# Checkpoints
SAVE_BEST = True
