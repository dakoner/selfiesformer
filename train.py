import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from functools import partial
import sys
import json
import os

import config
from selfies_tokenizer import SELFIES_Tokenizer
from dataset import SELFIES_Dataset, collate_fn
from model import TransformerAutoencoder

def load_selfies_from_file(filepath):
    """Loads a list of SELFIES from a file, one per line."""
    try:
        with open(filepath, 'r') as f:
            selfies_list = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(selfies_list)} SELFIES from {filepath}")
        return selfies_list
    except FileNotFoundError:
        print(f"❌ Error: The file '{filepath}' was not found.")
        sys.exit(1)

def run_epoch(model, dataloader, criterion, device, optimizer=None):
    """Runs one epoch of training or evaluation."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_correct_sequences, total_correct_tokens, total_tokens, total_samples = 0, 0, 0, 0, 0
    
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in dataloader:
            batch = batch.to(device)
            tgt_input, tgt_output = batch[:, :-1], batch[:, 1:]

            if is_train: optimizer.zero_grad()
            output = model(src=batch, tgt=tgt_input)
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
            
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            predictions = torch.argmax(output, dim=-1)
            total_correct_sequences += torch.all(predictions == tgt_output, dim=1).sum().item()
            total_samples += batch.size(0)
            non_pad_mask = (tgt_output != model.pad_idx)
            total_correct_tokens += ((predictions == tgt_output) & non_pad_mask).sum().item()
            total_tokens += non_pad_mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    seq_accuracy = (total_correct_sequences / total_samples) * 100 if total_samples > 0 else 0
    token_accuracy = (total_correct_tokens / total_tokens) * 100 if total_tokens > 0 else 0
    
    return avg_loss, seq_accuracy, token_accuracy

def main():
    print(f"Using device: {config.DEVICE}")

    # --- Check for Checkpoint ---
    start_epoch = 1
    tokenizer = SELFIES_Tokenizer()
    model = None
    optimizer = None

    if os.path.exists(config.MODEL_PATH) and os.path.exists(config.TOKENIZER_PATH):
        print(f"✅ Found existing checkpoint at '{config.MODEL_PATH}'")
        # 1. Load tokenizer
        with open(config.TOKENIZER_PATH, 'r') as f:
            symbol_to_idx = json.load(f)
        tokenizer.symbol_to_idx = symbol_to_idx
        tokenizer.idx_to_symbol = {i: s for s, i in symbol_to_idx.items()}
        print(f"Tokenizer loaded with {tokenizer.vocab_size} symbols.")

        # 2. Instantiate model and optimizer
        model = TransformerAutoencoder(
            vocab_size=tokenizer.vocab_size, d_model=config.D_MODEL, nhead=config.NHEAD,
            num_encoder_layers=config.NUM_ENCODER_LAYERS, num_decoder_layers=config.NUM_DECODER_LAYERS,
            dim_feedforward=config.DIM_FEEDFORWARD, dropout=config.DROPOUT, pad_idx=tokenizer.pad_idx
        ).to(config.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

        # 3. Load model and optimizer state
        checkpoint = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
        print(checkpoint.keys())
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Model and optimizer state loaded. Resuming training from epoch {start_epoch}.")
    
    else:
        print("🔎 No checkpoint found, starting from scratch.")

    # --- Data Preparation (or re-loading for dataset object) ---
    raw_selfies = load_selfies_from_file(config.DATA_FILE)
    if not tokenizer.vocab_size > 0: # Build vocab only if not loaded
        if len(raw_selfies) < 100:
            print("Dataset is small, duplicating it 100x for this demonstration.")
            raw_selfies *= 100
        tokenizer.build_vocab(raw_selfies)

    full_dataset = SELFIES_Dataset(raw_selfies, tokenizer)
    print("1")
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    custom_collate = partial(collate_fn, pad_idx=tokenizer.pad_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
    val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate)

    # --- Instantiate Model and Optimizer if not loaded from checkpoint ---
    if model is None:
        model = TransformerAutoencoder(
            vocab_size=tokenizer.vocab_size, d_model=config.D_MODEL, nhead=config.NHEAD,
            num_encoder_layers=config.NUM_ENCODER_LAYERS, num_decoder_layers=config.NUM_DECODER_LAYERS,
            dim_feedforward=config.DIM_FEEDFORWARD, dropout=config.DROPOUT, pad_idx=tokenizer.pad_idx
        ).to(config.DEVICE)
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)
    # --- Training Loop ---
    print("\nStarting training... 🚀 (Press Ctrl+C to interrupt and save)")
    try:
        for epoch in range(start_epoch, config.EPOCHS + 1):
            train_loss, train_seq_acc, train_token_acc = run_epoch(model, train_dataloader, criterion, config.DEVICE, optimizer)
            val_loss, val_seq_acc, val_token_acc = run_epoch(model, val_dataloader, criterion, config.DEVICE)
            
            print(f"Epoch: {epoch:02d} | Train Loss: {train_loss:.4f}, Token Acc: {train_token_acc:.2f}% | Val Loss: {val_loss:.4f}, Token Acc: {val_token_acc:.2f}%")
        
        print("\nTraining finished completely.")
    except KeyboardInterrupt:
        print("\n\n🛑 Training interrupted by user.")
    finally:
        # --- Save Model, Optimizer, and Tokenizer ---
        # The 'epoch' variable will exist if the loop started
        if 'epoch' in locals():
            print(f"\nSaving checkpoint from epoch {epoch}...")
            
            # Save model and optimizer state
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, config.MODEL_PATH)
            
            # Save tokenizer vocabulary
            with open(config.TOKENIZER_PATH, 'w') as f:
                json.dump(tokenizer.symbol_to_idx, f, indent=4)
                
            print("\n✅ Checkpoint saved. You can now run reconstruct.py or resume training later.")

if __name__ == '__main__':
    main()