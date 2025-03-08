import os
import random
from math import log

import torch
import torch.nn.functional as F
import jieba
from torch.xpu import device

from SongIambicsGeneration.utils import fix_poem_rhythm, \
    format_poem, check_and_fix_poem_line, find_model_files, load_model, safe_log
from my_config import *


def nucleus_sampling(probs, p=0.9):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)

    # **边界检查**
    valid_indices = (cumulative_probs > p).nonzero(as_tuple=True)[0]
    if valid_indices.numel() == 0:
        return sorted_indices[0].item()  # **如果 `cumulative_probs` 为空，返回 Top-1**

    cutoff_idx = valid_indices[0].item()
    sampled_index = sorted_indices[:cutoff_idx + 1]

    if sampled_index.numel() == 0:
        return sorted_indices[0].item()

    return sampled_index[torch.randint(len(sampled_index), (1,))].item()


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
                next_log_prob = safe_log(top_probs[i].item())  # 计算对数概率
                # 长度归一化
                length_penalized_log_prob = log_prob + next_log_prob / ((len(seq) + 1) ** length_penalty)

                new_candidates.append((seq + [next_token], length_penalized_log_prob))
        # 按对数概率排序，选择 top-k 候选序列
        new_candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = new_candidates[:beam_size]

    # 返回最优的候选序列
    best_sequence = candidates[0][0]
    return best_sequence


def simple_punishment(word_counts, idx2word, probs, punishment=0.1):
    for i in range(len(probs)):
        word = idx2word[i]
        if word in word_counts and word_counts[word] > 1:
            probs[i] *= punishment  # **降低已出现单词的概率**
    probs /= probs.sum()
    return probs


def nucleus_sampling_punishment(word_counts, idx2word, probs, punishment=0.05):
    # 使用核采样方式抑制重复单词
    repeated_tokens = [idx for idx, word in enumerate(idx2word) if word_counts.get(word, 0) > 1]

    if repeated_tokens:
        probs[repeated_tokens] *= 0.05  # **极大降低已出现单词的概率**

    probs /= probs.sum()
    return probs


def get_predicted_token(logits, beam_size, probs, top_p):
    predicted_token = (
        beam_search_v3(logits, beam_size=beam_size)[-1]
        if random.random() < 0.5
        else nucleus_sampling(probs, p=top_p)
    )
    return predicted_token


def generate_poetry_next_word_LSTM(model, input_token, hidden, cell, encoder_outputs,
                                   word_counts, idx2word, temperature, top_p, beam_size):
    eps = 1e-6
    with torch.no_grad():
        output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)
        logits = output.squeeze(0) / max(temperature, eps)
        probs = F.softmax(logits, dim=-1)

        probs = simple_punishment(word_counts, idx2word, probs) \
            if random.random() < 0.5 else nucleus_sampling_punishment(word_counts, idx2word, probs)

    predicted_token = get_predicted_token(logits, beam_size, probs, top_p)
    return predicted_token, hidden, cell, probs


def generate_poetry_next_word_GRU(model, input_token, hidden, encoder_outputs,
                                  word_counts, idx2word, temperature, top_p, beam_size):
    eps = 1e-6
    with torch.no_grad():
        output, hidden = model.decoder(input_token, hidden, encoder_outputs)
        logits = output.squeeze(0) / max(temperature, eps)
        probs = F.softmax(logits, dim=-1)

        probs = simple_punishment(word_counts, idx2word, probs) \
            if random.random() < 0.5 else nucleus_sampling_punishment(word_counts, idx2word, probs)

    predicted_token = get_predicted_token(logits, beam_size, probs, top_p)

    return predicted_token, hidden, probs


def generate_poetry_next_word(model_type, model, input_token, hidden, cell, encoder_outputs,
                              word_counts, idx2word, temperature, top_p, beam_size):
    with torch.amp.autocast(device_type="cuda", enabled=True):
        if model_type == "LSTM":
            return generate_poetry_next_word_LSTM(model, input_token, hidden, cell, encoder_outputs,
                                                  word_counts, idx2word, temperature, top_p, beam_size)
        elif model_type == "GRU":
            return generate_poetry_next_word_GRU(model, input_token, hidden, encoder_outputs,
                                                 word_counts, idx2word, temperature, top_p, beam_size)
        else:
            raise ValueError("Invalid model type: {}".format(model_type))


def generate_poetry_next_word_plus(model_type, model, input_token, hidden, cell, encoder_outputs,
                                   word_counts, idx2word, temperature, top_p, beam_size):
    with torch.no_grad():
        if model_type == "LSTM":
            output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)
        elif model_type == "GRU":
            output, hidden = model.decoder(input_token, hidden, encoder_outputs)
        else:
            raise ValueError("Invalid model type: {}".format(model_type))

        logits = output.squeeze(0) / temperature
        probs = F.softmax(logits, dim=-1)

        # probs = simple_punishment(word_counts, idx2word, probs) \
        #     if random.random() < 0.5 else nucleus_sampling_punishment(word_counts, idx2word, probs)
        probs = simple_punishment(word_counts, idx2word, probs)

    predicted_token = get_predicted_token(logits, beam_size, probs, top_p)

    if model_type == "LSTM":
        return predicted_token, hidden, cell, probs
    elif model_type == "GRU":
        return predicted_token, hidden, probs
    else:
        raise ValueError("Invalid model type: {}".format(model_type))


def generate_poetry_line_by_line(model_type, model, tokenizer, device, keyword_phrase,
                                 sentence_length, total_line, temperature, top_p, beam_size):
    word2idx, idx2word = tokenizer
    keywords = list(jieba.cut(keyword_phrase))
    keyword_indices = [word2idx.get(word, word2idx["<UNK>"]) for word in keywords]
    src = torch.tensor(keyword_indices, device=device).unsqueeze(0)
    generated_poem = keywords.copy()
    word_counts = {}  # 统计已出现单词，防止重复
    hidden, cell = None, None
    with torch.no_grad():
        if model_type == "LSTM":
            encoder_outputs, hidden, cell = model.encoder(src)
        elif model_type == "GRU":
            encoder_outputs, hidden = model.encoder(src)
        else:
            raise ValueError("Invalid model type: {}".format(model_type))

    input_token = src[:, -1]
    for current_row in range(total_line):
        current_sentence = []
        for i in range(sentence_length):
            if model_type == "LSTM":
                predicted_token, hidden, cell, probs = generate_poetry_next_word_plus(
                    model_type, model, input_token, hidden, cell, encoder_outputs,
                    word_counts, idx2word, temperature, top_p, beam_size
                )
            elif model_type == "GRU":
                predicted_token, hidden, probs = generate_poetry_next_word_plus(
                    model_type, model, input_token, hidden, cell, encoder_outputs,
                    word_counts, idx2word, temperature, top_p, beam_size
                )
            else:
                raise ValueError("Invalid model type: {}".format(model_type))

            filtered_topk = [
                idx for idx in torch.topk(probs, k=15).indices.tolist()
                if idx2word[idx] not in SPECIAL_TKOKENS
            ]
            if not filtered_topk:
                filtered_topk = [word2idx.get(UNK_TOKEN, 0)]

            while idx2word[predicted_token] in SPECIAL_TKOKENS:
                predicted_token = random.choice(filtered_topk)

            last_word_idx = word2idx.get(current_sentence[-1]) if current_sentence else None
            while last_word_idx == predicted_token or idx2word[predicted_token] in SPECIAL_TKOKENS:
                predicted_token = random.choice(
                    torch.topk(probs, k=15).indices.tolist()) if random.random() > 0.25 else last_word_idx

            word_counts[idx2word[predicted_token]] = word_counts.get(idx2word[predicted_token], 0) + 1
            current_sentence.append(idx2word[predicted_token])
            input_token = torch.tensor([predicted_token], device=device)

        generated_poem.append("".join(current_sentence))

    return generated_poem


def generate_poetry_v1(model_type, model, tokenizer, device, keyword_phrase, sentence_length=7,
                       total_line=4, temperature=0.8, top_p=0.9, beam_size=4):
    model.eval()
    if isinstance(keyword_phrase, list):
        keyword_phrase = "".join(keyword_phrase)
    poem = generate_poetry_line_by_line(model_type, model, tokenizer, device, keyword_phrase,
                                        sentence_length, total_line, temperature, top_p, beam_size)
    total_line = check_and_fix_poem_line(total_line)
    return format_poem(
        sentence_pre_line=sentence_length,
        total_lines=total_line,
        poem=poem if isinstance(poem, str) else "".join(poem))


def generate_poetry_line_by_line_v2(model_type, model, tokenizer, device, keyword_phrase,
                                    sentence_length, total_line, temperature, top_p, beam_size):
    word2idx, idx2word = tokenizer
    keywords = list(jieba.cut(keyword_phrase))

    keyword_indices = [word2idx.get(word, word2idx["<UNK>"]) for word in keywords]
    src = torch.tensor(keyword_indices, device=device).unsqueeze(0)
    generated_poem = []
    word_counts = {}
    use_keywords = len(keywords) >= total_line
    hidden, cell = None, None
    with torch.no_grad():
        if model_type == "LSTM":
            encoder_outputs, hidden, cell = model.encoder(src)
        elif model_type == "GRU":
            encoder_outputs, hidden = model.encoder(src)
        else:
            raise ValueError("Invalid model type: {}".format(model_type))

    for current_line in range(total_line):

        if use_keywords:
            keyword_index = current_line % len(keyword_indices)
        else:
            keyword_index = current_line if current_line < len(keyword_indices) else None

        if keyword_index is not None:
            first_word = idx2word.get(keyword_indices[keyword_index], keywords[keyword_index])
            if first_word in SPECIAL_TKOKENS:
                first_word = keywords[keyword_index]
        else:
            first_word = None

        current_sentence = [first_word] if first_word else []

        if first_word:
            input_token = torch.tensor([keyword_indices[keyword_index]], device=device)
        else:
            input_token = torch.tensor([word2idx["<SOS>"]], device=device)

        for i in range(len(first_word) if first_word else 0, sentence_length):
            if model_type == "LSTM":
                predicted_token, hidden, cell, probs = generate_poetry_next_word_plus(
                    model_type, model, input_token, hidden, cell, encoder_outputs,
                    word_counts, idx2word, temperature, top_p, beam_size
                )
            elif model_type == "GRU":
                predicted_token, hidden, probs = generate_poetry_next_word_plus(
                    model_type, model, input_token, hidden, cell, encoder_outputs,
                    word_counts, idx2word, temperature, top_p, beam_size
                )
            else:
                raise ValueError("Invalid model type: {}".format(model_type))

            filtered_topk = [
                idx for idx in torch.topk(probs, k=15).indices.tolist()
                if idx2word[idx] not in SPECIAL_TKOKENS
            ]
            if not filtered_topk:
                filtered_topk = [word2idx.get(UNK_TOKEN, 0)]

            while idx2word[predicted_token] in SPECIAL_TKOKENS:
                predicted_token = random.choice(filtered_topk)

            last_word_idx = word2idx.get(current_sentence[-1]) if current_sentence else None
            while last_word_idx == predicted_token or idx2word[predicted_token] in SPECIAL_TKOKENS:
                predicted_token = random.choice(
                    torch.topk(probs, k=15).indices.tolist()) if random.random() > 0.25 else last_word_idx

            word_counts[idx2word[predicted_token]] = word_counts.get(idx2word[predicted_token], 0) + 1

            current_sentence.append(idx2word[predicted_token])
            input_token = torch.tensor([predicted_token], device=device)

        generated_poem.append("".join(current_sentence))

    return "".join(generated_poem)


def generate_poetry_v2(model_type, model, tokenizer, device, keyword_phrase, sentence_length=7,
                       total_line=4, temperature=0.8, top_p=0.9, beam_size=4):
    model.eval()
    if isinstance(keyword_phrase, list):
        keyword_phrase = "".join(keyword_phrase)
    total_line = check_and_fix_poem_line(total_line)
    poem = generate_poetry_line_by_line_v2(model_type, model, tokenizer, device, keyword_phrase,
                                           sentence_length, total_line, temperature, top_p, beam_size)

    return format_poem(
        sentence_pre_line=sentence_length,
        total_lines=total_line,
        poem=poem if isinstance(poem, str) else "".join(poem))


def generate_poetry(model_type, epoch_range, timestamp, model, tokenizer, device, keywords, sentence_length=7,
                    temperature=0.8,
                    top_p=0.9, beam_size=4):
    model_dir = "SavedModels"
    model_files = find_model_files(model_dir, epoch_range, model_type, timestamp)

    for idx, filepath in enumerate(model_files):
        loaded_model = load_model(model, filepath, device)
        if loaded_model is None:
            continue

        poem = generate_poetry_v2(
            model_type=model_type,
            model=loaded_model,
            tokenizer=tokenizer,
            device=device,
            keyword_phrase="".join(keywords),
            sentence_length=sentence_length,
            total_line=5,
            temperature=temperature,
            beam_size=beam_size,
            top_p=top_p
        )
        # poem = generate_acrostic_poetry(
        #     model_type=model_type,
        #     model=loaded_model,
        #     tokenizer=tokenizer,
        #     device=device,
        #     words=keywords,
        #     sentence_length=sentence_length,
        #     temperature=temperature,
        #     top_p=top_p,
        #     beam_size=beam_size
        # )
        # logging.info(f"NO.{idx} 生成的诗歌: {poem}")
        print(f"NO.{idx} 生成的诗歌: {poem}")


def impl_generate_acrostic_poetry(model_type, model, tokenizer, device, word_chars, sentence_length=7,
                                  temperature=0.8, top_p=0.9, beam_size=4):
    word2idx, idx2word = tokenizer

    word_indices = [word2idx.get(word, word2idx["<UNK>"]) for word in word_chars]
    src = torch.tensor(word_indices).unsqueeze(0).to(device)

    generated_poem = []
    word_counts = {}
    hidden, cell = None, None
    with torch.no_grad():
        if model_type == "LSTM":
            encoder_outputs, hidden, cell = model.encoder(src)
        elif model_type == "GRU":
            encoder_outputs, hidden = model.encoder(src)
        else:
            raise ValueError("Invalid model type: {}".format(model_type))

    for i in range(len(word_chars)):
        current_sentence = [word_chars[i]]
        input_token = torch.tensor([word_indices[i]]).to(device)

        while len(current_sentence) < sentence_length:

            if model_type == "LSTM":
                predicted_token, hidden, cell, probs = generate_poetry_next_word(
                    model_type, model, input_token, hidden, cell, encoder_outputs,
                    word_counts, idx2word, temperature, top_p, beam_size
                )
            elif model_type == "GRU":
                predicted_token, hidden, probs = generate_poetry_next_word(
                    model_type, model, input_token, hidden, cell, encoder_outputs,
                    word_counts, idx2word, temperature, top_p, beam_size
                )
            else:
                raise ValueError("Invalid model type: {}".format(model_type))

            filtered_topk = [
                idx for idx in torch.topk(probs, k=15).indices.tolist()
                if idx2word[idx] not in SPECIAL_TKOKENS
            ]
            if not filtered_topk:
                filtered_topk = [word2idx.get(UNK_TOKEN, 0)]

            while idx2word[predicted_token] in SPECIAL_TKOKENS:
                predicted_token = random.choice(filtered_topk)

            last_word_idx = word2idx.get(current_sentence[-1]) if current_sentence else None
            while last_word_idx == predicted_token or idx2word[predicted_token] in SPECIAL_TKOKENS:
                predicted_token = random.choice(
                    torch.topk(probs, k=5).indices.tolist()) if random.random() > 0.1 else last_word_idx

            word_counts[idx2word[predicted_token]] = word_counts.get(idx2word[predicted_token], 0)
            current_sentence.append(idx2word[predicted_token])
            input_token = torch.tensor([predicted_token]).to(device)

        generated_poem.append("".join(current_sentence))

    return "".join(generated_poem)


def generate_acrostic_poetry(model_type, model, tokenizer, device, words, sentence_length=7,
                             temperature=0.8, top_p=0.9, beam_size=2):
    model.eval()

    if isinstance(words, str):
        words = [words]

    if not isinstance(words, list):
        raise TypeError('words must be a list')

    if len(words) == 0:
        raise ValueError('words must not be empty')

    if len(words) > 4:
        words = words[:4]

    word_chars = [char for word in words for char in word]
    poem = impl_generate_acrostic_poetry(model_type, model, tokenizer, device, word_chars, sentence_length, temperature,
                                         top_p,
                                         beam_size)

    return format_poem(
        sentence_pre_line=sentence_length,
        total_lines=len(word_chars),
        poem=poem if isinstance(poem, str) else "".join(poem))
