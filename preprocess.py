import os
import json
import jieba
import torch
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from my_config import *


def replace_special_characters(text: str):
    new_text = []
    cnt = 0
    for char in text:
        if char == '|':
            if cnt % 2 == 0:
                new_text.append('，')
            else:
                new_text.append('。')
            cnt += 1
        else:
            new_text.append(char)
    return new_text

def process_CCPC_data(src_path, trg_path):
    with open(trg_path, "w", encoding="utf-8") as write_file:
        song_poetry = []
        with open(src_path, "r", encoding="utf-8") as read_file:
            for line in read_file:
                if line.strip():
                    json_object = json.loads(line)
                    text = json_object["content"]
                    text = replace_special_characters("".join(text))
                    song_poetry.append(text)

        for poetry in song_poetry:
            write_file.write("".join(poetry) + "\n")

def load_data(path):
    with open(path,"r",encoding="utf-8") as f:
        lines = f.readlines()
    return [line.strip() for line in lines if len(line.strip()) > 0]

def clean_text(text):
    text = text.replace("，", "").replace("。", "").replace("？", "").replace("！", "")
    text = text.replace("；", "").replace("、", "").replace("：", "").replace("（", "").replace("）", "")
    return text

def tokenize(text):
    return list(text)

def tokenize_with_jieba(text):
    return list(jieba.cut(text))

def build_vocab(texts,min_freq=3):
    word_freq = {}
    for text in texts:
        for word in text:
            word_freq[word] = word_freq.get(word,0) + 1

    vocab = SPECIAL_TKOKENS + [word for word, freq in word_freq.items() if freq >= min_freq]
    word2idx =  {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(word2idx, f, ensure_ascii=False)

    print(f"Vocabulary build complete，vocab size：{len(vocab)}")
    return word2idx, idx2word

def text_to_indices(text, word2idx):
    indices = [word2idx.get(SOS_TOKEN, 0)]
    indices += [word2idx.get(word, word2idx[UNK_TOKEN]) for word in text]
    indices.append(word2idx.get(EOS_TOKEN, 0))

    if len(indices) > MAX_LENGTH:
        indices = indices[:MAX_LENGTH]
    return indices

def preprocess_data():
    print("Start preprocessing data...")
    process_CCPC_data(CCPC_DATA_PATH, DATA_PATH)
    texts = load_data(DATA_PATH)
    texts = [clean_text(text) for text in texts]
    tokenized_texts = [tokenize(text) for text in texts]

    with open(SENTENCES_PATH, "w", encoding="utf-8") as f:
        for text in tokenized_texts:
            f.write(" ".join(text) + "\n")

    word2idx, idx2word = build_vocab(tokenized_texts)
    indexed_texts = [text_to_indices(text, word2idx) for text in tokenized_texts]
    print(f"vocab size: {len(word2idx)}")

    torch.save(indexed_texts, PROCESSED_PATH)
    print("Data preprocessing complete.")

if __name__ == "__main__":
    preprocess_data()
