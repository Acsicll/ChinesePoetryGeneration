import time

import gensim
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from gensim.models import Word2Vec
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from SongIambicsGeneration.dataset import get_dataloader, load_vocab, load_sentences
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
from models.attention import Attention
import random


def train(model,iterator,optimizer,criterion,clip,device):
    model.train()
    epoch_loss = 0
    scaler = torch.amp.GradScaler(enabled=True)
    for i, batch in enumerate(iterator):
        src, trg = batch
        src, trg = src.to(device, non_blocking=True), trg.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda', enabled=True):
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# def get_Word2vec(word2idx, sentences, embedding_dim, device):
#     model = Word2Vec(sentences, vector_size=embedding_dim, window=5, min_count=1, sg=0)
#
#     vocab_size = len(word2idx)
#     embedding_matrix = np.zeros((vocab_size, embedding_dim))
#     for word, idx in word2idx.items():
#         if word in model.wv:
#             embedding_matrix[idx] = model.wv[word]
#         else:
#             embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))  # 未知词随机初始化
#
#     embedding_matrix[word2idx["<UNK>"]] = np.mean(embedding_matrix, axis=0)
#     return nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32)).to(device)

def to_train(model,train_loader,optimizer,criterion,clip,device):
    cudnn.benchmark = True
    cudnn.enabled = True

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    #for epoch in range(20):
    epoch = 20
    while epoch < 30:
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, clip, device)
        scheduler.step(train_loss)
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(int(end_time - start_time), 60)
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Time: {epoch_mins}m {epoch_secs}s')
        torch.save(model.state_dict(), f'./SavedModels/seq2seq_{epoch+1}.pt')
        print(f"Model saved as models./SavedModels/seq2seq_{epoch+1}.pt")
        epoch += 1

    print("Training complete!")
