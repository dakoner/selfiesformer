import torch
import json
import argparse
import sys

import config # Import the configuration file
from selfies_tokenizer import SELFIES_Tokenizer
from model import TransformerAutoencoder

def main():
    # --- Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Reconstruct a SELFIES string using a trained Transformer model.")
    parser.add_argument(
        "selfies_string", 
        type=str, 
        help="The SELFIES string to reconstruct. E.g., '[C][=C][C][=C][C][=C][Ring1][Branch1_1]'"
    )
    args = parser.parse_args()
    
    print(f"Using device: {config.DEVICE}")

    # --- Load Tokenizer ---
    print(f"Loading tokenizer vocabulary from {config.TOKENIZER_PATH}...")
    try:
        with open(config.TOKENIZER_PATH, 'r') as f:
            symbol_to_idx = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Tokenizer file not found at '{config.TOKENIZER_PATH}'. Please run train.py first.", file=sys.stderr)
        sys.exit(1)

    tokenizer = SELFIES_Tokenizer()
    tokenizer.symbol_to_idx = symbol_to_idx
    tokenizer.idx_to_symbol = {i: s for s, i in symbol_to_idx.items()}
    
    # --- Load Model ---
    print(f"Loading model from {config.MODEL_PATH}...")
    model = TransformerAutoencoder(
        vocab_size=tokenizer.vocab_size,
        d_model=config.D_MODEL,
        nhead=config.NHEAD,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT,
        pad_idx=tokenizer.pad_idx
    ).to(config.DEVICE)
    
    try:
        # CORRECTED: Load the checkpoint dictionary and then the model's state_dict
        checkpoint = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at '{config.MODEL_PATH}'. Please run train.py first.", file=sys.stderr)
        sys.exit(1)
    except KeyError:
        print(f"❌ Error: 'model_state_dict' key not found in the checkpoint. The saved file may be corrupted or from an older version.", file=sys.stderr)
        sys.exit(1)
        
    model.eval()

    # --- Perform Reconstruction ---
    print(f"\n--- Reconstruction Test ---")
    test_selfies_str = args.selfies_string
    print(f"Original SELFIES:      {test_selfies_str}")

    tokenized_input = tokenizer.tokenize(test_selfies_str)
    if not tokenized_input:
        print("❌ Error: Could not tokenize the input SELFIES string.", file=sys.stderr)
        sys.exit(1)
        
    src_tensor = torch.tensor(tokenized_input, dtype=torch.long).unsqueeze(0).to(config.DEVICE)
    
    generated_tokens_tensor = model.generate(
        src=src_tensor, 
        sos_idx=tokenizer.sos_idx, 
        eos_idx=tokenizer.eos_idx,
        max_len=src_tensor.shape[1] + 10
    )
    
    generated_tokens = generated_tokens_tensor.squeeze(0).cpu().tolist()
    reconstructed_selfies = tokenizer.detokenize(generated_tokens)
    
    print(f"Reconstructed SELFIES: {reconstructed_selfies}")
    print(f"Reconstruction Successful: {test_selfies_str == reconstructed_selfies}")

if __name__ == '__main__':
    main()