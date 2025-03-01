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

def nucleus_sampling(probs, p=0.9):
    """
    实现 Top-P (Nucleus Sampling) 采样：
    - 选择累积概率达到 `p` 的 top-n 词汇
    - 然后在其中随机采样
    """
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 找到第一个累积概率 > p 的位置
    cutoff_idx = (cumulative_probs > p).nonzero(as_tuple=True)[0][0]

    # 只保留前 `cutoff_idx` 个最高概率的词
    top_p_probs = sorted_probs[:cutoff_idx + 1]
    top_p_indices = sorted_indices[:cutoff_idx + 1]

    # 归一化后采样
    top_p_probs /= top_p_probs.sum()
    sampled_index = torch.multinomial(top_p_probs, 1).item()
    return top_p_indices[sampled_index].item()

def beam_search(logits, beam_size=3):
    """Beam Search 选取最佳候选词 """
    top_probs, top_indices = torch.topk(F.softmax(logits, dim=-1), k=beam_size)
    return top_indices[random.randint(0, beam_size - 1)].item()  # 随机从前3个选一个

def beam_search_v2(logits, beam_size=3, max_len=10, eos_token_id=2):
    candidates = [([], 0.0)]  # 每个候选序列是一个元组：(序列, 对数概率)

    for step in range(max_len):
        new_candidates = []

        for seq, log_prob in candidates:
            if len(seq) > 0 and seq[-1] == eos_token_id:
                # 如果序列已经结束，直接保留
                new_candidates.append((seq, log_prob))
                continue

            # 获取当前时间步的 logits
            #print("logits:",logits.shape)
            current_logits = logits.squeeze(0) # [1, vocab_size]
            probs = F.softmax(current_logits, dim=-1)  # [1, vocab_size]

            # 选择 top-k 候选词
            top_probs, top_indices = torch.topk(probs, k=beam_size)  # [batch_size, beam_size]

            for i in range(beam_size):
                next_token = top_indices[i].item()  # 选择第 i 个候选词
                next_log_prob = log(top_probs[i].item())  # 计算对数概率

                new_seq = seq + [next_token]
                new_log_prob = log_prob + next_log_prob

                new_candidates.append((new_seq, new_log_prob))

        # 按对数概率排序，选择 top-k 候选序列
        new_candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = new_candidates[:beam_size]

    # 返回最优的候选序列
    best_sequence = candidates[0][0]
    return best_sequence

def beam_search_v3(logits, beam_size=4, max_len=10, eos_token_id=2, length_penalty=0.6):

    candidates = [([], 0.0)]  # 每个候选序列是一个元组：(序列, 对数概率)

    for step in range(max_len):
        new_candidates = []

        for seq, log_prob in candidates:
            if len(seq) > 0 and seq[-1] == eos_token_id:
                # 如果序列已经结束，直接保留
                new_candidates.append((seq, log_prob))
                continue

            # 获取当前时间步的 logits
            current_logits = logits.squeeze(0)  # [1, vocab_size]
            probs = F.softmax(current_logits, dim=-1)  # [1, vocab_size]

            # 选择 top-k 候选词
            top_probs, top_indices = torch.topk(probs, k=beam_size)  # [batch_size, beam_size]

            for i in range(beam_size):
                next_token = top_indices[i].item()  # 选择第 i 个候选词
                next_log_prob = log(top_probs[i].item())  # 计算对数概率

                # 长度归一化
                length_penalized_log_prob = log_prob + next_log_prob / (len(seq) + 1) ** length_penalty

                new_seq = seq + [next_token]
                new_candidates.append((new_seq, length_penalized_log_prob))

        # 按对数概率排序，选择 top-k 候选序列
        new_candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = new_candidates[:beam_size]

    # 返回最优的候选序列
    best_sequence = candidates[0][0]
    return best_sequence



def fix_poem_rhythm(poem):
    """
    🎵 后处理：检查韵律（可选）
    - 确保偶数行对仗
    - 调整句尾字韵脚
    """
    # TODO: 这里可以基于音韵库进行优化，目前只是简单返回
    return poem


if __name__=="__main__":
    check_vocab()
