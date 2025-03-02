import json
import random
from math import log

import torch.nn.functional as F
import torch

from my_config import *

def check_vocab():
    """
    Check if the given vocabulary is valid.
    """
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        word2idx = json.load(f)
    idx2word = {v: k for k, v in word2idx.items()}

    print("word2idx sample:",list(word2idx.items()))
    print("idx2word sample:",list(idx2word.items()))






def fix_poem_rhythm(poem):
    """
     后处理：检查韵律（可选）
    - 确保偶数行对仗
    - 调整句尾字韵脚
    """
    # TODO: 这里可以基于音韵库进行优化，目前只是简单返回
    return poem

def format_poem(poem : str, sentence_pre_line, total_lines):
    result = []
    for index, char in enumerate(poem):
        if index == sentence_pre_line * total_lines:
            break
        result.append(char)
        if (index + 1) % sentence_pre_line == 0:
            result.append('，' if (index // sentence_pre_line) % 2 == 0 else '。')
    return "".join(result)

def check_and_fix_poem_line(total_line:int):
    if not isinstance(total_line,int):
        raise ValueError("total_line must be int")

    if total_line < 4:
        total_line = 4
    if total_line % 2 == 1:
        total_line += 1
    return total_line

if __name__=="__main__":
    check_vocab()
