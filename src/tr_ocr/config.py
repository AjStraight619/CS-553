# Hyperparameters for training
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 5e-5
MAX_LENGTH = 256  # Max tokenized length for LaTeX formulas

# Model and dataset configurations
MODEL_NAME = "microsoft/trocr-base-stage1"
DATASET_NAME = "OleehyO/latex-formulas"
DATASET_CONFIG = "cleaned_formulas"

# Dataset split chunk size
CHUNK_SIZE = 50000  # Number of examples per chunk
START_INDEX = 0  # Start index for the current chunk
END_INDEX = CHUNK_SIZE  # End index for the current chunk

# Output directories
SAVE_DIR = "preprocessed_datasets"  # Directory to store preprocessed chunks
OUTPUT_DIR = "output_model"  # Directory to save the final trained model