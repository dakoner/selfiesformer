import selfies as sf

class SELFIES_Tokenizer:
    """
    A tokenizer for SELFIES strings. Builds a vocabulary and provides methods
    for converting between SELFIES strings and integer sequences.
    """
    def __init__(self):
        self.symbol_to_idx = {}
        self.idx_to_symbol = {}
        self.specials = ['<pad>', '<sos>', '<eos>', '<unk>']

    def build_vocab(self, selfies_list):
        """
        Builds the vocabulary from a list of SELFIES strings.

        Args:
            selfies_list (list[str]): A list of SELFIES strings to build the vocabulary from.
        """
        print("Building vocabulary...")
        all_symbols = set()
        for s in selfies_list:
            try:
                symbols = list(sf.split_selfies(s))
                all_symbols.update(symbols)
            except (sf.DecoderError, sf.EncoderError) as e:
                print(f"Warning: Could not process SELFIES '{s}'. Error: {e}")
                continue
            except TypeError:
                pass
                import pdb; pdb.set_trace()
        vocab = self.specials + sorted(list(all_symbols))
        self.symbol_to_idx = {s: i for i, s in enumerate(vocab)}
        self.idx_to_symbol = {i: s for i, s in enumerate(vocab)}
        print(f"Vocabulary built. Size: {self.vocab_size}")

    def tokenize(self, selfies_string):
        """
        Tokenizes a single SELFIES string into a list of integers.
        Includes <sos> and <eos> tokens.

        Args:
            selfies_string (str): The SELFIES string to tokenize.

        Returns:
            list[int]: A list of integer tokens.
        """
        try:
            symbols = list(sf.split_selfies(selfies_string))
            tokens = [self.symbol_to_idx['<sos>']]
            tokens.extend([self.symbol_to_idx.get(s, self.symbol_to_idx['<unk>']) for s in symbols])
            tokens.append(self.symbol_to_idx['<eos>'])
            return tokens
        except (sf.DecoderError, sf.EncoderError):
            return None # Or handle error appropriately

    def detokenize(self, tokens, clean_up=True):
        """
        Converts a list of integer tokens back into a SELFIES string.

        Args:
            tokens (list[int]): The list of integer tokens.
            clean_up (bool): If True, removes special tokens before joining.

        Returns:
            str: The reconstructed SELFIES string.
        """
        symbols = [self.idx_to_symbol[t] for t in tokens]
        if clean_up:
            cleaned_symbols = []
            for s in symbols:
                if s == '<eos>':
                    break
                if s not in self.specials:
                    cleaned_symbols.append(s)
            symbols = cleaned_symbols
        return "".join(symbols)

    @property
    def vocab_size(self):
        return len(self.symbol_to_idx)

    @property
    def pad_idx(self):
        return self.symbol_to_idx['<pad>']

    @property
    def sos_idx(self):
        return self.symbol_to_idx['<sos>']

    @property
    def eos_idx(self):
        return self.symbol_to_idx['<eos>']