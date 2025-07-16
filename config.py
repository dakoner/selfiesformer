import torch

# --- Configuration & File Paths ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DATA_FILE = "selfies.txt"
DATA_FILE = "dataJ_250k_rndm_zinc_drugs_clean.sf.txt"
#DATA_FILE = "zinc_drugs_10000.txt"
MODEL_PATH = "transformer_autoencoder.pth"
TOKENIZER_PATH = "tokenizer_vocab.json"


# --- Model Hyperparameters ---
D_MODEL = 128
NHEAD = 4
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT = 0.1


# --- Training Hyperparameters ---
EPOCHS = 50
BATCH_SIZE = 1024
LEARNING_RATE = 0.0001