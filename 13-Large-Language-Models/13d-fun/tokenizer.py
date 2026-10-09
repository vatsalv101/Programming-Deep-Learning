from collections import Counter


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")

class BytePairTokenizer:
    
    """Small byte-level BPE tokenizer."""

    def __init__(self, special_tokens=SPECIAL_TOKENS):

        """
        Creates a byte-level BPE tokenizer.

        The class should provide the following interface:
        - attributes: pad_id, bos_id, eos_id, unk_id, vocab_size
        - methods: train, encode, decode, token_to_bytes

        Args:
            special_tokens (tuple): Tokens reserved for padding, sequence boundaries and unknown values.
        """

        self.special_tokens = special_tokens
        self.special_to_id = {}
        self.id_to_special = {}

        for index, token in enumerate(special_tokens):
            self.special_to_id[token] = index
            self.id_to_special[index] = token

        self.pad_id = self.special_to_id["<pad>"]
        self.bos_id = self.special_to_id["<bos>"]
        self.eos_id = self.special_to_id["<eos>"]
        self.unk_id = self.special_to_id["<unk>"]

        self.byte_offset = len(special_tokens)
        self.first_merge_id = self.byte_offset + 256
        self.merges = []
        self.vocab_size = self.first_merge_id

    def train(self, texts, vocab_size=1024, min_frequency=2):

        """
        Trains the byte-level BPE tokenizer.

        Args:
            texts (iterable): Texts used to learn frequent byte-pair merges.
            vocab_size (int): Maximum tokenizer vocabulary size.
            min_frequency (int): Minimum pair count required for a merge.

        Returns:
            self (BytePairTokenizer): Trained tokenizer.
        """

        min_vocab_size = self.byte_offset + 256
        if vocab_size < min_vocab_size:
            raise ValueError("vocab_size must be at least " + str(min_vocab_size))

        sequences = []
        for text in texts:
            if text:
                sequences.append(self.text_to_byte_tokens(text))

        self.merges = []

        for new_token_id in range(self.first_merge_id, vocab_size):
            pair_counts = self.count_pairs(sequences)
            if len(pair_counts) == 0:
                break

            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < min_frequency:
                break

            for index in range(len(sequences)):
                sequences[index] = self.replace_pair(sequences[index], best_pair, new_token_id)

            self.merges.append(best_pair)

        self.vocab_size = self.first_merge_id + len(self.merges)

        return self

    def encode(self, text, add_bos=False, add_eos=False):

        """
        Encodes text into BPE token ids.

        Args:
            text (str): Input text.
            add_bos (bool): If True, prepend the beginning-of-sequence token.
            add_eos (bool): If True, append the end-of-sequence token.

        Returns:
            token_ids (list): Token ids.
        """

        token_ids = self.text_to_byte_tokens(text)

        for merge_index, pair in enumerate(self.merges):
            merged_id = self.first_merge_id + merge_index
            token_ids = self.replace_pair(token_ids, pair, merged_id)

        if add_bos:
            token_ids = [self.bos_id] + token_ids
        if add_eos:
            token_ids = token_ids + [self.eos_id]

        return token_ids

    def decode(self, token_ids, skip_special_tokens=True):

        """
        Decodes BPE token ids back into text.

        Args:
            token_ids (list): Token ids.
            skip_special_tokens (bool): If True, ignore special tokens.

        Returns:
            text (str): Decoded text.
        """

        byte_values = []

        for token_id in token_ids:
            token_id = int(token_id)
            if token_id in self.id_to_special:
                if not skip_special_tokens:
                    byte_values += list(self.id_to_special[token_id].encode("utf-8"))
            else:
                byte_values += self.token_to_bytes(token_id)

        return bytes(byte_values).decode("utf-8", errors="replace")

    def token_to_bytes(self, token_id):

        """
        Converts one token id back to the bytes it represents.

        Args:
            token_id (int): Token id.

        Returns:
            byte_values (list): Byte values represented by the token.
        """

        token_id = int(token_id)

        if self.byte_offset <= token_id < self.first_merge_id:
            return [token_id - self.byte_offset]

        merge_index = token_id - self.first_merge_id
        if 0 <= merge_index < len(self.merges):
            left_id, right_id = self.merges[merge_index]
            return self.token_to_bytes(left_id) + self.token_to_bytes(right_id)

        return list(self.id_to_special[self.unk_id].encode("utf-8"))

    def text_to_byte_tokens(self, text):

        """
        Converts text to byte token ids.

        Args:
            text (str): Input text.

        Returns:
            token_ids (list): Byte token ids.
        """

        token_ids = []
        for byte in text.encode("utf-8"):
            token_ids.append(byte + self.byte_offset)

        return token_ids

    def count_pairs(self, sequences):

        """
        Counts neighboring token pairs in a list of token sequences.

        Args:
            sequences (list): List of token-id sequences.

        Returns:
            pair_counts (Counter): Counts for neighboring token pairs.
        """

        pair_counts = Counter()

        for sequence in sequences:
            for index in range(len(sequence) - 1):
                pair = (sequence[index], sequence[index + 1])
                pair_counts[pair] += 1

        return pair_counts

    def replace_pair(self, token_ids, pair, merged_id):

        """
        Replaces non-overlapping occurrences of one pair by one merged token id.

        Args:
            token_ids (list): Input token ids.
            pair (tuple): Pair to replace.
            merged_id (int): New token id.

        Returns:
            merged (list): Token ids after replacement.
        """

        merged = []
        index = 0

        while index < len(token_ids):
            has_pair = index + 1 < len(token_ids)
            if has_pair and token_ids[index] == pair[0] and token_ids[index + 1] == pair[1]:
                merged.append(merged_id)
                index += 2
            else:
                merged.append(token_ids[index])
                index += 1

        return merged
    

