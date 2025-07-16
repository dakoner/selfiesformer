import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input embeddings.
    From the "Attention Is All You Need" paper.
    """
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
        # Assumes x is [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerAutoencoder(nn.Module):
    """
    A Transformer-based autoencoder.
    """
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers,
                 num_decoder_layers, dim_feedforward, dropout, pad_idx):
        super(TransformerAutoencoder, self).__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def _generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        """
        Forward pass for training with teacher forcing.
        Args:
            src (Tensor): [batch_size, src_len]
            tgt (Tensor): [batch_size, tgt_len]
        """
        src_key_padding_mask = (src == self.pad_idx)
        tgt_key_padding_mask = (tgt == self.pad_idx)
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1), src.device)

        # Permute from [batch, seq] to [seq, batch] for embedding/pos_encoder
        src_emb = self.pos_encoder(self.embedding(src).permute(1,0,2) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt).permute(1,0,2) * math.sqrt(self.d_model))
        
        # Permute back to [batch, seq] for batch_first=True transformer layers
        src_emb = src_emb.permute(1,0,2)
        tgt_emb = tgt_emb.permute(1,0,2)

        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        output = self.transformer_decoder(tgt_emb, memory,
                                          tgt_mask=tgt_mask,
                                          tgt_key_padding_mask=tgt_key_padding_mask,
                                          memory_key_padding_mask=src_key_padding_mask)

        return self.fc_out(output)

    def encode(self, src):
        self.eval()
        with torch.no_grad():
            src_key_padding_mask = (src == self.pad_idx)
            src_emb = self.pos_encoder(self.embedding(src).permute(1,0,2) * math.sqrt(self.d_model))
            src_emb = src_emb.permute(1,0,2)
            memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        return memory

    def generate(self, src, sos_idx, eos_idx, max_len=100):
        """
        Autoregressive generation for inference.
        """
        self.eval()
        with torch.no_grad():
            memory = self.encode(src)
            # Start with <sos> token
            ys = torch.ones(src.size(0), 1).fill_(sos_idx).long().to(src.device)
            for _ in range(max_len - 1):
                # Prepare masks and embeddings for decoder
                tgt_key_padding_mask = (ys == self.pad_idx)
                tgt_mask = self._generate_square_subsequent_mask(ys.size(1), src.device)
                
                tgt_emb = self.pos_encoder(self.embedding(ys).permute(1,0,2) * math.sqrt(self.d_model))
                tgt_emb = tgt_emb.permute(1,0,2)
                
                memory_key_padding_mask = (src == self.pad_idx)
                
                out = self.transformer_decoder(tgt_emb, memory,
                                               tgt_mask=tgt_mask,
                                               tgt_key_padding_mask=tgt_key_padding_mask,
                                               memory_key_padding_mask=memory_key_padding_mask)
                
                # Get the last predicted token
                last_token_logits = self.fc_out(out[:, -1, :])
                next_token = torch.argmax(last_token_logits, dim=1)
                
                # Append predicted token to the sequence
                ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
                
                # Stop if all sequences in batch have generated <eos>
                if (next_token == eos_idx).all():
                    break
        return ys