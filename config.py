import torch

class Config:
    # Paths
    DATA_DIR = '/Users/samswitz/handwritten-names/'
    TRAIN_CSV = 'written_name_train_v2.csv'
    VAL_CSV = 'written_name_validation_v2.csv'
    TEST_CSV = 'written_name_test_v2.csv'
    TRAIN_IMG_DIR = 'train_v2/train'
    VAL_IMG_DIR = 'validation_v2/validation'
    TEST_IMG_DIR = 'test_v2/test'

    # Model checkpoints
    CHECKPOINT_DIR = 'checkpoints'

    # Image settings
    IMG_HEIGHT = 128
    IMG_WIDTH = 512
    NUM_CHANNELS = 1

    # Character vocabulary
    # Based on the dataset, using uppercase letters, space, and digits
    CHARS = ' ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    BLANK_LABEL = len(CHARS)  # CTC blank token
    NUM_CLASSES = len(CHARS) + 1  # +1 for blank

    # Model selection
    USE_TRANSFORMER = True  # Toggle between CRNN and Transformer

    # CRNN Model architecture (commented out - not in use)
    # CNN_OUTPUT_CHANNELS = 512
    # RNN_HIDDEN_SIZE = 256
    # RNN_NUM_LAYERS = 2
    # RNN_DROPOUT = 0.2

    # Transformer architecture settings
    PATCH_SIZE = 64  # Size of each patch (64x64)
    EMBED_DIM = 256  # Embedding dimension
    TRANSFORMER_LAYERS = 6  # Number of encoder layers
    TRANSFORMER_HEADS = 8   # Number of attention heads
    TRANSFORMER_DIM_FF = 1024  # Feed-forward dimension
    TRANSFORMER_DROPOUT = 0.1

    # Training hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # DataLoader
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # Early stopping
    PATIENCE = 5

    # Logging
    PRINT_FREQ = 100
    SAVE_FREQ = 1  # Save every N epochs
