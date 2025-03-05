import json
import os
import random
import time
from math import log

import jieba
import networkx as nx
import torch.nn.functional as F
import torch
import opencc
from sqlalchemy.sql.operators import truediv

from my_config import *


def generate_model_name(task, dataset, model_type, lr, batch_size, epoch, val_loss):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{task}_{dataset}_{model_type}_lr{lr}_bs{batch_size}_ep{epoch}_vl{val_loss:.3f}_{timestamp}.pt"


def check_vocab():
    """
    Check if the given vocabulary is valid.
    """
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        word2idx = json.load(f)
    idx2word = {v: k for k, v in word2idx.items()}

    print("word2idx sample:", list(word2idx.items()))
    print("idx2word sample:", list(idx2word.items()))


def fix_poem_rhythm(poem):
    """
     后处理：检查韵律（可选）
    - 确保偶数行对仗
    - 调整句尾字韵脚
    """
    # TODO: 这里可以基于音韵库进行优化，目前只是简单返回
    return poem


def format_poem(poem: str, sentence_pre_line, total_lines):
    result = []
    for index, char in enumerate(poem):
        if index == sentence_pre_line * total_lines:
            break
        result.append(char)
        if (index + 1) % sentence_pre_line == 0:
            result.append('，' if (index // sentence_pre_line) % 2 == 0 else '。')
    return "".join(result)


def check_and_fix_poem_line(total_line: int):
    if not isinstance(total_line, int):
        raise ValueError("total_line must be int")

    if total_line < 4:
        total_line = 4
    if total_line % 2 == 1:
        total_line += 1
    return total_line


def tranditional_chinese_to_simpilfied_chinese(src_file_path, dst_file_path):
    converter = opencc.OpenCC('t2s.json')
    try:
        with open(src_file_path, 'r', encoding='utf-8') as read_file, \
                open(dst_file_path, 'w', encoding='utf-8') as write_file:
            lines = read_file.readlines()
            converted_lines = []

            for line in lines:
                if any('\u4e00' <= char <= '\u9fa5' for char in line):
                    converted_line = converter.convert(line)
                    converted_lines.append(converted_line)
                else:
                    converted_lines.append(line)

            write_file.write(converter.convert(line))
    except FileNotFoundError as e:
        print(f"File error: {e}")
        print("Tranditional Chinese to Simplified Chinese Done!")
    except UnicodeDecodeError as e:
        print(f"An error occurred : {e}")

    print("Tranditional Chinese to Simplified Chinese Done!")


def extract_keywords_textrank_jieba(poems, top_k=1, window_size=3):
    """
    使用 jieba 进行分词，并用 TextRank 提取关键词。
    :param poems: 诗歌列表，每首诗为一行字符串
    :param top_k: 每首诗提取多少个关键词
    :param window_size: 计算共现关系的窗口大小
    :return: 关键词列表（与 poems 对应）
    """
    keywords = []
    for poem in poems:
        words = jieba.lcut(poem)  # 使用 jieba 进行分词

        # 过滤停用词（如果需要，可以扩展停用词表）
        words = [word for word in words if word not in STOP_WORDS and len(word) > 1]

        if not words:
            keywords.append("<UNK>")
            continue

        # 构建词图
        graph = nx.Graph()
        graph.add_nodes_from(words)

        # 计算共现权重（增加窗口大小）
        for i, word in enumerate(words):
            for j in range(i + 1, min(i + window_size, len(words))):
                if words[j] != word:
                    if graph.has_edge(word, words[j]):
                        graph[word][words[j]]["weight"] += 1.0
                    else:
                        graph.add_edge(word, words[j], weight=1.0)

        # 计算 PageRank 得分
        scores = nx.pagerank(graph, weight="weight")
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 取最高得分的关键词
        top_keywords = [word for word, _ in sorted_words[:top_k]]
        keywords.append(top_keywords[0] if top_keywords else "<UNK>")

    return keywords


def check_classify_dir(directory: str):
    for name in POETRY_THEMES.keys():
        file_name = f"{directory}/{name}.txt"
        if not os.path.exists(file_name):
            try:
                open(file_name, 'w', encoding='utf-8').close()
            except IOError as e:
                print(e)
    print("Check classify dir done!")


def classify_poetry_by_theme(poetry: str, classified_dict: dict):
    keywords = extract_keywords_textrank_jieba([poetry], top_k=1, window_size=3)
    for keyword in keywords:
        for theme, theme_words in POETRY_THEMES.items():
            if keyword in theme_words or any (char in theme_words for char in poetry):
                classified_dict[theme].append([poetry])


def classify_poetry(directory, src_file_path, themes=POETRY_THEMES):
    if not isinstance(directory, str):
        raise ValueError("directory must be str")
    elif not os.path.exists(directory):
        raise ValueError("input file does not exist")
    elif os.path.isdir(directory):
        check_classify_dir(directory)
    else:
        raise ValueError("directory must be str or path")

    classified_dict = {theme: [] for theme in themes.keys()}

    with open(src_file_path, 'r', encoding='utf-8') as read_file:
        poetries = read_file.readlines()
        for poetry in poetries:
            classify_poetry_by_theme(poetry, classified_dict)

    for theme, poetries in classified_dict.items():
        with open(f"{directory}/{theme}.txt", 'a', encoding='utf-8') as write_file:
            unique_poetries = set(tuple(poetry) for poetry in poetries)
            for poetry in unique_poetries:
                write_file.write("".join(poetry))

    print("Classify poetry done!")


if __name__ == "__main__":
    # check_vocab()

    classify_poetry(CLASSIFIED_DIR_PATH, DATA_PATH)
