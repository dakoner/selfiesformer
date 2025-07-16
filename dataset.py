import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class SELFIES_Dataset(Dataset):
    """
    PyTorch Dataset for handling SELFIES strings.
    """
    def __init__(self, selfies_list, tokenizer):
        """
        Args:
            selfies_list (list[str]): List of SELFIES strings.
            tokenizer (SELFIES_Tokenizer): The tokenizer instance.
        """
        self.tokenizer = tokenizer
        self.tokenized_selfies = []
        for s in selfies_list:
            tokens = self.tokenizer.tokenize(s)
            if tokens: # Ensure tokenization was successful
                 self.tokenized_selfies.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self):
        return len(self.tokenized_selfies)

    def __getitem__(self, idx):
        return self.tokenized_selfies[idx]

def collate_fn(batch, pad_idx):
    """
    Collator function to pad sequences in a batch to the same length.
    """
    return pad_sequence(batch, batch_first=True, padding_value=pad_idx)