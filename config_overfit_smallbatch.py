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
    CNN_OUTPUT_CHANNELS = 512
    RNN_HIDDEN_SIZE = 256
    RNN_NUM_LAYERS = 2
    RNN_DROPOUT = 0.0

    # # Transformer architecture settings
    PATCH_SIZE = 16  # Size of each patch (16x16) - reduced from 64 for better resolution
    EMBED_DIM = 128
    TRANSFORMER_LAYERS = 4
    TRANSFORMER_HEADS = 4
    TRANSFORMER_DIM_FF = 1024
    TRANSFORMER_DROPOUT = 0.1

    # Training hyperparameters
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-5

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # DataLoader
    NUM_WORKERS = 1
    PIN_MEMORY = False

    # Early stopping
    PATIENCE = 5

    # Logging
    PRINT_FREQ = 100
    SAVE_FREQ = 1  # Save every N epochs
