import os

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import json
from my_config import *


# 1️⃣ **加载词表**
def load_vocab():
    with open(SIMPLE_VOCAB_PATH, "r", encoding="utf-8") as f:
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
        self.data = torch.load(SIMPLE_PROCESSED_PATH)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq =  torch.tensor(self.data[idx], dtype=torch.long)
        src = seq[:-1]
        trg = seq[1:]
        return src, trg

# 3️⃣ **创建 DataLoader**
def get_dataloader(batch_size=32, train_ratio = 0.8):
    dataset = SongDataset()
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count() - 2,
        pin_memory=True,
        prefetch_factor=4) # [batch_sie, seq_len]
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count() - 2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4)

    return train_loader, val_loader


if __name__ == "__main__":
    word2idx, idx2word = load_vocab()
    train_loader, val_loader = get_dataloader(batch_size=32)
    for i,batch in enumerate(train_loader):
        src, trg = batch
        print(f"Sample {i + 1} src:", src[0].tolist())  # 查看索引序列
        print("Decoded src:", "".join([idx2word[i] for i in src[0].tolist()]))  # 转换回汉字
        print("=====")
        if i == 10:
            break