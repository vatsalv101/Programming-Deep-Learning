from collections import Counter


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")


class BytePairTokenizer:
    
    """Small byte-level BPE tokenizer."""

    def __init__(self, special_tokens=SPECIAL_TOKENS):

        """
        Creates a byte-level BPE tokenizer.

        Args:
            special_tokens (tuple): Tokens reserved for padding, sequence boundaries and unknown values.
        """
        
    ################################################################
    # TODO
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

    def train(self, texts, vocab_size = 1024, min_frequency = 2):
        if vocab_size < self.vocab_size:
            raise ValueError("error")

        seq = []

        for text in texts:
            if text:
                seq.append(self.byte_to_token(text))

        self.merges = []

        for new_merge_id in range(self.first_merge_id, vocab_size):
            count_pairs = self.counter(seq)

            if len(count_pairs) == 0:
                break            

            best_pair, best_count = count_pairs.most_common(1)[0]

            if best_count < min_frequency:
                break

            for i in range(len(seq)):
                seq[i] = self.replace_pair(new_merge_id, best_pair, seq[i])

            self.merges.append(best_pair)

        self.vocab_size = self.byte_offset + 256 + len(self.merges)

        return self
    

    def replace_pair(self, new_merge_id, pair, text):
        i = 0
        l = []
        while i < len(text):
            has_pair = i + 1 < len(text)
            if has_pair and text[i] == pair[0] and text[i+1] == pair[1]:
                l.append(new_merge_id)
                i+=2
            else:
                l.append(text[i])
                i+=1
        return l


    def counter(self, texts):
        count_pairs = Counter()
        for text in texts:
            for i in range(len(text)-1):
                pair = (text[i], text[i+1])
                count_pairs[pair] += 1
        return count_pairs


    def byte_to_token(self, text):
        l = []
        for t in text.encode('utf-8'):
            l.append(t + self.byte_offset)
        return l

    def encode(self, text, add_bos = False, add_eos = False):
        seq = self.byte_to_token(text)

        for i in range(len(self.merges)):
            merge_id = self.first_merge_id + i
            seq = self.replace_pair(merge_id, self.merges[i], seq)

        if add_bos:
            seq = [self.bos_id] + seq
        if add_eos:
            seq = seq + [self.eos_id]
        return seq

    def decode(self, token_ids, skip_special_tokens = True):
        l = []
        for id in token_ids:
            id = int(id)
            if 0 <= id <= self.byte_offset:
                if not skip_special_tokens:
                    l += list(self.id_to_special[id].encode('utf-8'))
            else:
                l += self.token_to_bytes(id)
        return bytes(l).decode('utf-8', errors = "replace")

    def token_to_bytes(self, id):
        id = int(id)
        if self.byte_offset <= id < self.first_merge_id:
            return [id - self.byte_offset]
        
        elif self.first_merge_id <= id < self.vocab_size:
            left, right = self.merges[id - self.first_merge_id]
            return self.token_to_bytes(left) + self.token_to_bytes(right)

        return list(self.id_to_token(self.unk_id).encode('utf-8'))        
    ################################################################
    

