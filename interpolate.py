import torch
import json
import argparse
import sys
import selfies as sf
import numpy as np

import config
from selfies_tokenizer import SELFIES_Tokenizer
from model import TransformerAutoencoder

def main():
    # --- Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Interpolate between two SELFIES strings in latent space.")
    parser.add_argument("selfies1", type=str, help="The starting SELFIES string.")
    parser.add_argument("selfies2", type=str, help="The ending SELFIES string.")
    parser.add_argument("--steps", type=int, default=10, help="The number of interpolation steps.")
    args = parser.parse_args()
    
    print(f"Using device: {config.DEVICE}")

    # --- Load Tokenizer and Model ---
    try:
        with open(config.TOKENIZER_PATH, 'r') as f:
            symbol_to_idx = json.load(f)
        tokenizer = SELFIES_Tokenizer()
        tokenizer.symbol_to_idx = symbol_to_idx
        tokenizer.idx_to_symbol = {i: s for s, i in symbol_to_idx.items()}
    except FileNotFoundError:
        print(f"❌ Error: Tokenizer file not found at '{config.TOKENIZER_PATH}'. Please run train.py first.", file=sys.stderr)
        sys.exit(1)

    model = TransformerAutoencoder(
        vocab_size=tokenizer.vocab_size, d_model=config.D_MODEL, nhead=config.NHEAD,
        num_encoder_layers=config.NUM_ENCODER_LAYERS, num_decoder_layers=config.NUM_DECODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD, dropout=config.DROPOUT, pad_idx=tokenizer.pad_idx
    ).to(config.DEVICE)
    
    try:
        checkpoint = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at '{config.MODEL_PATH}'. Please run train.py first.", file=sys.stderr)
        sys.exit(1)
        
    model.eval()

    # --- Encode the two SELFIES strings ---
    s1_tokens = torch.tensor(tokenizer.tokenize(args.selfies1), dtype=torch.long).unsqueeze(0).to(config.DEVICE)
    s2_tokens = torch.tensor(tokenizer.tokenize(args.selfies2), dtype=torch.long).unsqueeze(0).to(config.DEVICE)
    
    memory1 = model.encode(s1_tokens)
    memory2 = model.encode(s2_tokens)
    
    # --- Create a fixed-size latent vector by averaging over the sequence length ---
    # This is a common way to get a sentence/sequence embedding from a Transformer
    latent1 = torch.mean(memory1, dim=1, keepdim=True)
    latent2 = torch.mean(memory2, dim=1, keepdim=True)
    
    print("\n--- Latent Space Interpolation ---")
    
    # --- Interpolate and Decode ---
    for i in range(args.steps + 1):
        alpha = i / args.steps
        
        # Linearly interpolate between the two latent vectors
        inter_latent = torch.lerp(latent1, latent2, alpha)
        
        # The decoder expects a 'memory' tensor of shape [batch, seq_len, dim]
        # Our interpolated vector is [1, 1, dim], which works perfectly.
        generated_tokens_tensor = model.generate_from_latent(
            memory=inter_latent,
            sos_idx=tokenizer.sos_idx,
            eos_idx=tokenizer.eos_idx,
            max_len=150 # Set a generous max length for generated molecules
        )
        
        generated_tokens = generated_tokens_tensor.squeeze(0).cpu().tolist()
        reconstructed_selfies = tokenizer.detokenize(generated_tokens)
        
        # Check if the generated SELFIES string is valid
        is_valid = sf.decoder(reconstructed_selfies) is not None
        valid_marker = "✅" if is_valid else "❌"
        
        print(f"Step {i:02d} (alpha={alpha:.2f}): {reconstructed_selfies} ({valid_marker})")

if __name__ == '__main__':
    main()