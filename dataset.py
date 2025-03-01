import torch
from torch.utils.data import Dataset, DataLoader
import json
from my_config import *


# 1️⃣ **加载词表**
def load_vocab():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        word2idx = json.load(f)
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word

def load_sentences():
    sentences = []
    with open(SENTENCES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            word_list = line.strip().split()
            sentences.append(word_list)
    return sentences

# 2️⃣ **创建 Dataset**
class SongDataset(Dataset):
    def __init__(self):
        self.data = torch.load(PROCESSED_PATH)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq =  torch.tensor(self.data[idx], dtype=torch.long)
        src = seq[:-1]
        trg = seq[1:]
        return src, trg

# 3️⃣ **创建 DataLoader**
def get_dataloader(batch_size=32):
    dataset = SongDataset()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2) # [batch_sie, seq_len]

if __name__ == "__main__":
    word2idx, idx2word = load_vocab()
    dataloader = get_dataloader(batch_size=32)
    for i,batch in enumerate(dataloader):
        src, trg = batch
        print(f"Sample {i + 1} src:", src[0].tolist())  # 查看索引序列
        print("Decoded src:", "".join([idx2word[i] for i in src[0].tolist()]))  # 转换回汉字
        print("=====")
        if i == 10:
            break