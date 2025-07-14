import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import selfies as sf
import random
import math
from selfies_data import selfies_data
# 1. Data Preparation and Tokenization

class SELFIES_Dataset(Dataset):
    def __init__(self, selfies_list, vocab, max_len):
        self.selfies_list = selfies_list
        self.vocab = vocab
        self.max_len = max_len
        self.token_to_idx = {token: i for i, token in enumerate(vocab)}
        self.idx_to_token = {i: token for i, token in enumerate(vocab)}
        self.pad_token_idx = self.token_to_idx['<pad>']

    def __len__(self):
        return len(self.selfies_list)

    def __getitem__(self, idx):
        selfie = self.selfies_list[idx]
        tokens = list(sf.split_selfies(selfie))
        
        # Add start and end tokens
        tokens = ['<sos>'] + tokens + ['<eos>']
        
        # Pad sequence
        padded_tokens = tokens + ['<pad>'] * (self.max_len - len(tokens))
        
        # Convert to indices
        indexed_tokens = [self.token_to_idx.get(token, self.token_to_idx['<unk>']) for token in padded_tokens]
        
        return torch.tensor(indexed_tokens)

# 2. Model Architecture

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerAutoencoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length):
        super(TransformerAutoencoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_length)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        memory = self.transformer_encoder(src_emb)
        output = self.transformer_decoder(tgt_emb, memory)
        
        return self.fc_out(output)

# 3. Training and Evaluation

def train(model, dataloader, optimizer, criterion, vocab_size, device):
    model.train()
    total_loss = 0
    for data in dataloader:
        # Move data to the selected device (GPU/CPU)
        src = data.to(device)
        
        # For autoencoder, the target is the same as the source, but shifted
        tgt_input = src[:, :-1]
        tgt_output = src[:, 1:]

        optimizer.zero_grad()
        output = model(src, tgt_input)
        
        loss = criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def main():
    # --- Hyperparameters ---
    D_MODEL = 128
    NHEAD = 4
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    DIM_FEEDFORWARD = 512
    MAX_SEQ_LENGTH = 512
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001

    # --- Setup CUDA Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} 🚀")

    # # --- Data ---
    # # Example SELFIES data (replace with your dataset)
    # smiles_data = ["CC(C)C", "c1ccccc1", "CCO", "CNC(=O)C", "CC(=O)Oc1ccccc1C(=O)O"]
    # selfies_data = [sf.encoder(s) for s in smiles_data]

    # --- Vocabulary ---
    all_tokens = set(['<pad>', '<unk>', '<sos>', '<eos>'])
    for selfie in selfies_data:
        all_tokens.update(list(sf.split_selfies(selfie)))
    vocab = sorted(list(all_tokens))
    vocab_size = len(vocab)
    
    # --- Dataloader ---
    dataset = SELFIES_Dataset(selfies_data, vocab, MAX_SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- Model Initialization ---
    model = TransformerAutoencoder(vocab_size, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, MAX_SEQ_LENGTH).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.token_to_idx['<pad>'])

    # --- Training Loop ---
    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, dataloader, optimizer, criterion, vocab_size, device)
        print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.4f}')
    torch.save(model.state_dict(), "model.save.pth")

    # --- Inference Example ---
    model.eval()
    with torch.no_grad():
        test_selfie = random.choice(selfies_data)
        print(f"\nOriginal SELFIES: {test_selfie}")
        
        test_dataset = SELFIES_Dataset([test_selfie], vocab, MAX_SEQ_LENGTH)
        test_loader = DataLoader(test_dataset, batch_size=1)
        
        # Move inference data to the device
        src = next(iter(test_loader)).to(device)

        # Start with the <sos> token, also on the correct device
        tgt_input = torch.tensor([[dataset.token_to_idx['<sos>']]]).to(device)

        for _ in range(MAX_SEQ_LENGTH - 1):
            output = model(src, tgt_input)
            next_token_logits = output[0, -1, :]
            next_token_idx = torch.argmax(next_token_logits).item()
            
            if next_token_idx == dataset.token_to_idx['<eos>']:
                break
            
            # Append the predicted token index to the target input
            next_token_tensor = torch.tensor([[next_token_idx]]).to(device)
            tgt_input = torch.cat([tgt_input, next_token_tensor], dim=1)

        # Retrieve indices from GPU to CPU for decoding
        reconstructed_indices = tgt_input.squeeze().cpu().tolist()
        reconstructed_tokens = [dataset.idx_to_token[idx] for idx in reconstructed_indices[1:]] # Exclude <sos>
        reconstructed_selfie = "".join(reconstructed_tokens)
        print(f"Reconstructed SELFIES: {reconstructed_selfie}")
        try:
            reconstructed_smiles = sf.decoder(reconstructed_selfie)
            print(f"Reconstructed SMILES: {reconstructed_smiles}")
        except sf.DecoderError:
            print("Could not decode reconstructed SELFIES.")


if __name__ == '__main__':
    print("a")
    main()