"""Character-level tokenizer shared by training and evaluation."""


class Tokenizer:
    def __init__(self, pad_id=1, unk_id=0):
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.string_to_id = {}
        self.id_to_string = {}
        # 0 and 1 are reserved for UNK and PAD by default.
        self.next_id = 2

    def build_vocab(self, texts):
        """Build the character vocabulary from training texts only."""
        for text in texts:
            for char in text:
                if char not in self.string_to_id:
                    self.string_to_id[char] = self.next_id
                    self.id_to_string[self.next_id] = char
                    self.next_id += 1

    def encode(self, text, max_len):
        """Encode, truncate and pad one text to exactly ``max_len`` ids."""
        output = []

        if text is not None:
            for char in text:
                output.append(self.string_to_id.get(char, self.unk_id))

        # An empty example must contain one visible token.  If it were all PAD,
        # every attention score would be masked and softmax would produce NaN.
        if not output:
            output = [self.unk_id]

        output = output[:max_len]
        output.extend([self.pad_id] * (max_len - len(output)))
        return output

    def get_vocab_size(self):
        return self.next_id

