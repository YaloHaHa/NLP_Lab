from pathlib import Path

# File Paths
ROOT_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"


SEQ_LEN = 128 # determined by the 95th percentile of review sequence length for LSTM input
BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_DIM = 256

LEARNING_RATE = 0.001
EPOCHS = 10