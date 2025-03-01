import random

import torch
import torch.nn.functional as F

from SongIambicsGeneration.utils import nucleus_sampling, beam_search, fix_poem_rhythm, beam_search_v3, beam_search_v2
from my_config import SPECIAL_TKOKENS

def generate_poem_temperature(model, tokenizer, device, temperature=1.0, max_length=32, start_text="<SOS>"):
    """
    通过 temperature 控制多样性
    """
    model.eval()
    word2idx, idx2word = tokenizer
    src = torch.tensor([word2idx.get(start_text, word2idx["<UNK>"])]).unsqueeze(0).to(device)
    generated_poem = [start_text]

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src)

    input_token = src.squeeze(0)

    for _ in range(max_length):
        with torch.no_grad():
            output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)
            logits = output.squeeze(0) / temperature
            probs = F.softmax(logits, dim=-1)
            predicted_token = torch.multinomial(probs, 1).item()

        if predicted_token == word2idx["<EOS>"]:
            break

        generated_poem.append(idx2word[predicted_token])
        input_token = torch.tensor([predicted_token]).to(device)

    return "".join(generated_poem).replace("<SOS>", "").replace("<EOS>", "")

import torch
import torch.nn.functional as F

def generate_poem_with_keywords(model, tokenizer, device, keywords, max_length=32, temperature=1.0):

    model.eval()
    word2idx, idx2word = tokenizer

    keyword_indices = [word2idx.get(word, word2idx["<UNK>"]) for word in keywords]
    src = torch.tensor(keyword_indices).unsqueeze(0).to(device)  # [1, len(keywords)]
    generated_poem = keywords.copy()

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src)

    input_token = src[:, -1]  # 以最后一个关键词作为起始 Token

    for _ in range(max_length - len(keywords)):  # 控制最大长度
        with torch.no_grad():
            output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)
            logits = output.squeeze(0) / temperature  # 采样温度
            probs = F.softmax(logits, dim=-1)
            predicted_token = torch.multinomial(probs, 1).item()  # 采样下一个 Token

        if predicted_token == word2idx["<EOS>"]:
            break

        if idx2word[predicted_token] == "<UNK>":
            continue  # 直接跳过这次生成

        generated_poem.append(idx2word[predicted_token])
        input_token = torch.tensor([predicted_token]).to(device)  # 更新输入

    return "".join(generated_poem)

import torch
import torch.nn.functional as F
import jieba

def generate_poem_with_jieba_keywords(model, tokenizer, device, keyword_phrase, max_length=32, temperature=1.0, top_k=5):
    model.eval()
    word2idx, idx2word = tokenizer

    if isinstance(keyword_phrase, list):
        keyword_phrase = "".join(keyword_phrase)
    keywords = list(jieba.cut(keyword_phrase))

    print(f"分词结果: {keywords}")  # 例如: ["春风", "又绿", "江南", "岸"]

    keyword_indices = [word2idx.get(word, word2idx["<UNK>"]) for word in keywords]
    src = torch.tensor(keyword_indices).unsqueeze(0).to(device)  # [1, len(keywords)]
    generated_poem = keywords.copy()

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src)

    input_token = src[:, -1]  # 以最后一个关键词作为起点

    for _ in range(max_length - len(keywords)):
        with torch.no_grad():
            output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)

            logits = output.squeeze(0) / temperature
            probs = F.softmax(logits, dim=-1)

            top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
            sampled_index = torch.multinomial(top_k_probs, 1).item()
            predicted_token = top_k_indices[sampled_index].item()

        if predicted_token == word2idx["<EOS>"]:
            if len(generated_poem) < max_length // 2:
                predicted_token = random.choice(top_k_indices.tolist())
            else:
                break
        # **避免生成 `<UNK>`**
        if idx2word[predicted_token] in SPECIAL_TKOKENS:
            continue  # 直接跳过 `<UNK>`

        generated_poem.append(idx2word[predicted_token])
        input_token = torch.tensor([predicted_token]).to(device)

    return "".join(generated_poem)

def generate_poem_with_jieba_keywords_v2(model, tokenizer, device, keyword_phrase, max_length=32, temperature=0.8, top_p=0.9):
    model.eval()
    word2idx, idx2word = tokenizer

    if isinstance(keyword_phrase, list):
        keyword_phrase = "".join(keyword_phrase)

    keywords = list(jieba.cut(keyword_phrase))
    print(f"🔹 分词结果: {keywords}")

    keyword_indices = [word2idx.get(word, word2idx["<UNK>"]) for word in keywords]
    src = torch.tensor(keyword_indices).unsqueeze(0).to(device)
    generated_poem = keywords.copy()

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src)

    input_token = src[:, -1]

    word_counts = {}  # **用于惩罚重复单词**

    for _ in range(max_length - len(keywords)):
        with torch.no_grad():
            output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)

            logits = output.squeeze(0) / temperature
            probs = F.softmax(logits, dim=-1)

            for i in range(len(probs)):
                word = idx2word[i]
                if word in word_counts:
                    probs[i] *= 0.5  # **降低已出现词的概率**

            predicted_token = nucleus_sampling(probs, p=top_p)

        if predicted_token == word2idx["<EOS>"]:
            if len(generated_poem) < max_length // 2:
                predicted_token = random.choice(torch.topk(probs, k=10).indices.tolist())
            else:
                break

        if idx2word[predicted_token] in SPECIAL_TKOKENS:
            continue

        word_counts[idx2word[predicted_token]] = word_counts.get(idx2word[predicted_token], 0) + 1

        generated_poem.append(idx2word[predicted_token])
        input_token = torch.tensor([predicted_token]).to(device)

    return "".join(generated_poem)

def generate_poem_with_keywords_v3(model, tokenizer, device, keyword_phrase, max_length=32, temperature=0.8, top_p=0.9, beam_size=3):
    model.eval()
    word2idx, idx2word = tokenizer

    if isinstance(keyword_phrase, list):
        keyword_phrase = "".join(keyword_phrase)
    keywords = list(jieba.cut(keyword_phrase))
    print(f"🔹 分词结果: {keywords}")

    keyword_indices = [word2idx.get(word, word2idx["<UNK>"]) for word in keywords]
    src = torch.tensor(keyword_indices).unsqueeze(0).to(device)
    generated_poem = keywords.copy()

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src)

    input_token = src[:, -1]
    word_counts = {}  # 统计已出现单词，防止重复

    #for _ in range(max_length - len(keywords)):
    while len(generated_poem) + 2 <= max_length:
        with torch.no_grad():
            output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)
            logits = output.squeeze(0) / temperature
            probs = F.softmax(logits, dim=-1)

            for i in range(len(probs)):
                word = idx2word[i]
                if word in word_counts:
                    probs[i] *= 0.2  # **降低已出现单词的概率**

            predicted_token = beam_search_v3(logits.unsqueeze(0), beam_size=beam_size)[-1]

        while predicted_token == word2idx["<EOS>"]:
            if len(generated_poem) < max_length:
                predicted_token = random.choice(torch.topk(probs, k=10).indices.tolist())
            else:
                break

        if idx2word[predicted_token] == "<UNK>":
            continue

        word_counts[idx2word[predicted_token]] = word_counts.get(idx2word[predicted_token], 0) + 1

        generated_poem.append(idx2word[predicted_token])
        input_token = torch.tensor([predicted_token]).to(device)

    generated_poem = fix_poem_rhythm(generated_poem)

    return "".join(generated_poem)