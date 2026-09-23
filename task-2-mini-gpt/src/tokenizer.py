# 测试：python -m src.tokenizer

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "train.txt"

def load_text(path=TRAIN_PATH):
    return Path(path).read_text(encoding="utf-8")

from collections import defaultdict

class BPETokenizer:
    def __init__(self):
        
        self.token_to_id = {}
        self.id_to_token = {}
        
        for i in range(256):
            token = bytes([i])

            self.token_to_id[token] = i
            self.id_to_token[i] = token
        self.merge = {}
    
    def merge_pair(self, ids, pair, new_id):
        new_ids = []
        i = 0
    
        while i < len(ids):
            if (
                i < len(ids) - 1
                and ids[i] == pair[0]
                and ids[i + 1] == pair[1]
            ):
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
    
        return new_ids
            
    def train(self, text, vocab_size):
        
        
        
        
        # 学 merge 规则
        ids = list(text.encode("utf-8"))
        
        new_id = 256
        
        
        while len(self.id_to_token) < vocab_size:
            counts = defaultdict(int)
            
            for i in range(len(ids)-1):
                pair = (ids[i], ids[i + 1])
                counts[pair] += 1
            if not counts:
                break
            best_pair = max(counts, key=counts.get)
                    
            new_id = 256 + len(self.merge)
            self.merge[best_pair] = new_id
            
            
            new_token = (
                            self.id_to_token[best_pair[0]]
                            + self.id_to_token[best_pair[1]]
                        )

            self.id_to_token[new_id] = new_token
            self.token_to_id[new_token] = new_id

            ids = self.merge_pair(ids, best_pair, new_id)
        
        
        
        
    def encode(self, text):
        ids = list(text.encode("utf-8"))

        for pair, new_id in self.merge.items():
            ids = self.merge_pair(ids, pair, new_id)
        
        return ids
    
    
    def decode(self, ids):
        raw = b"".join(
                        self.id_to_token[id_] for id_ in ids
                    )
        return raw.decode("utf-8")
    @property
    def vocab_size(self):
       return len(self.id_to_token) 
    def save(self, path):
        import json

        merges_list = []

        for pair, new_id in self.merge.items():
            left_id = pair[0]
            right_id = pair[1]

            merges_list.append([
                left_id,
                right_id,
                new_id
            ])

        data = {
            "merges": merges_list
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    
    @classmethod
    def from_pretrained(cls, path):
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls()

        for left_id, right_id, new_id in data["merges"]:

            pair = (left_id, right_id)

            tokenizer.merge[pair] = new_id

            new_token = (
                tokenizer.id_to_token[left_id]
                + tokenizer.id_to_token[right_id]
            )

            tokenizer.id_to_token[new_id] = new_token
            tokenizer.token_to_id[new_token] = new_id

        return tokenizer
        
if __name__ == "__main__":
    text = load_text()
    ids = list(text.encode("utf-8"))
    print("字符数：", len(text))
    print("text 结构：", type(text))
    print(ids[:100])