import os
import json
from collections import Counter
import re
import jieba
import numpy as np
import torch
from gensim.models import Word2Vec
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from SongIambicsGeneration.utils import tranditional_chinese_to_simpilfied_chinese
from my_config import *


def replace_special_characters(text: str) -> str:
    cnt = 0
    return "".join(
        "，" if (char == "|" and cnt % 2 == 0) else
        "。" if (char == "|" and cnt % 2 == 1) else
        char
        for char in text
    )


def process_CCPC_data(src_path, trg_path):
    with open(trg_path, "w", encoding="utf-8") as write_file, open(src_path, "r", encoding="utf-8") as read_file:
        for line in read_file:
            try:
                json_object = json.loads(line.strip())
                text = replace_special_characters("".join(json_object["content"]))
                write_file.write(text + "\n")
            except (json.JSONDecodeError, KeyError):
                print(f"⚠️ 跳过无效 JSON 行: {line[:30]}")


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [line.strip() for line in lines if len(line.strip()) > 0]


def clean_text(text: str) -> str:
    return re.sub(r"[，。？！；、：（）]", "", text)  # 一次性替换多个标点


def simple_tokenize(text):
    return list(text)


def tokenize_jieba_version(text):
    return [word for word in jieba.cut(text) if word not in STOP_WORDS]


def build_vocab(texts, min_freq=3, embedding_dim=256, vocab_path=SIMPLE_VOCAB_PATH,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    if not texts:
        raise ValueError("输入数据 texts 不能为空")
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            word2idx = json.load(f)
        idx2word = {idx: word for word, idx in word2idx.items()}
        print(f"Loaded existing vocabulary from {vocab_path}，vocab size：{len(word2idx)}")
        return word2idx, idx2word

    word_freq = Counter()
    for text in texts:
        word_freq.update(text)

    vocab = SPECIAL_TKOKENS + [word for word, freq in word_freq.items() if freq >= min_freq]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    with open(SIMPLE_VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(word2idx, f, ensure_ascii=False)

    print(f"Vocabulary build complete，vocab size：{len(vocab)}")

    vocab_size = len(vocab)
    embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size,embedding_dim))

    embedding_matrix[word2idx["<UNK>"]] = np.random.normal(scale=0.6, size=(embedding_dim,))
    embedding_matrix[word2idx["<SOS>"]] = np.mean(embedding_matrix, axis=0)
    embedding_matrix[word2idx["<EOS>"]] = np.mean(embedding_matrix, axis=0)
    embedding_matrix[word2idx["<PAD>"]] = np.zeros(embedding_dim)

    embedding_layer = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32)).to(device)
    return word2idx, idx2word, embedding_layer


def text_to_indices(text, word2idx, max_length=MAX_LENGTH):
    indices = [word2idx.get(SOS_TOKEN, 0)] + \
              [word2idx.get(word, word2idx[UNK_TOKEN]) for word in text] + \
              [word2idx.get(EOS_TOKEN, 0)]

    if len(indices) > max_length:
        indices = indices[:max_length]
    return indices


def build_vocab_with_Word2vec(sentences, embedding_dim=256,
                              special_tokens=None, vocab_path=WORD2VEC_VOCAB_PATH,
                              window=5, min_count=1, sg=0,
                              device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    if not sentences:
        raise ValueError("输入数据 sentences 不能为空")
    if special_tokens is None:
        special_tokens = SPECIAL_TKOKENS

    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            word2idx = json.load(f)
        idx2word = {idx: word for word, idx in word2idx.items()}
        print(f"Loaded existing vocabulary from {vocab_path}，vocab size：{len(word2idx)}")
        return word2idx, idx2word, None

    word2vec_model = Word2Vec(sentences, vector_size=embedding_dim, window=window, min_count=min_count, sg=sg,
                              workers=4)
    base_word2idx = word2vec_model.wv.key_to_index

    if len(base_word2idx) == 0:
        raise ValueError(f"min_count={min_count} 设置过高，未找到符合条件的单词")

    word2idx = {token: i for i, token in enumerate(special_tokens)}
    offset = len(special_tokens)

    word2idx.update({word: idx + offset for word, idx in base_word2idx.items()})
    idx2word = {idx: word for word, idx in word2idx.items()}

    vocab_size = len(word2idx)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, idx in base_word2idx.items():
        embedding_matrix[idx + offset] = word2vec_model.wv[word]

    embedding_matrix[word2idx["<UNK>"]] = np.random.normal(scale=0.6, size=(embedding_dim,))
    embedding_matrix[word2idx["<SOS>"]] = np.mean(embedding_matrix, axis=0)
    embedding_matrix[word2idx["<EOS>"]] = np.mean(embedding_matrix, axis=0)
    embedding_matrix[word2idx["<PAD>"]] = np.zeros(embedding_dim)

    with open(SIMPLE_VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(word2idx, f, ensure_ascii=False)

    embedding_layer = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32)).to(device)

    return word2idx, idx2word, embedding_layer


def preprocess_data(embedding_dim=256, build_vocab_mode: str = "simple_vocab"):
    global embedding_layer
    print("Start preprocessing data...")
    try:
        process_CCPC_data(CCPC_DATA_PATH, DATA_PATH)
        tranditional_chinese_to_simpilfied_chinese(DATA_PATH, T2S_DATA_PATH)
        texts = load_data(T2S_DATA_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    texts = [clean_text(text) for text in texts]
    tokenized_texts = [simple_tokenize(text) for text in texts if text != ""]

    with open(SENTENCES_PATH, "w", encoding="utf-8") as f:
        f.writelines([" ".join(text) + "\n" for text in tokenized_texts])

    indexed_texts_saving_path = ""
    word2idx, idx2word = None, None
    if build_vocab_mode == "pretrain_word2vec":
        word2idx, idx2word, embedding_layer = build_vocab_with_Word2vec(tokenized_texts, embedding_dim)
        indexed_texts_saving_path = WORD2VEC_PROCESSED_PATH
    elif build_vocab_mode == "simple_vocab":
        word2idx, idx2word, embedding_layer= build_vocab(tokenized_texts)
        indexed_texts_saving_path = SIMPLE_PROCESSED_PATH

    if word2idx is None or idx2word is None:
        raise ValueError("word2idx or idx2word is None")

    indexed_texts = [text_to_indices(text, word2idx) for text in tokenized_texts]

    print(indexed_texts_saving_path)
    torch.save(indexed_texts, indexed_texts_saving_path)
    print("Data preprocessing complete.")
    return (word2idx, idx2word), embedding_layer


if __name__ == "__main__":
    preprocess_data()
