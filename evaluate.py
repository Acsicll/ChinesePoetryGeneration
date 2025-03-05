import torch
from nltk.translate.bleu_score import sentence_bleu
from dataset import load_vocab
from SongIambicsGeneration.models.LSTM.encoder import Encoder
from SongIambicsGeneration.models.LSTM.decoder import Decoder
from SongIambicsGeneration.models.LSTM.seq2seq import Seq2Seq
from SongIambicsGeneration.models.attention import Attention

# 1️⃣ 加载词表
word2idx, idx2word = load_vocab()

# 2️⃣ 加载模型
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HIDDEN_DIM, N_LAYERS, ENC_DROPOUT).to(DEVICE)
attn = Attention(HIDDEN_DIM).to(DEVICE)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HIDDEN_DIM, N_LAYERS, DEC_DROPOUT, attn).to(DEVICE)
model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

model.load_state_dict(torch.load("./models/model_epoch20.pt"))
model.eval()

# 3️⃣ 评估 BLEU
def evaluate_bleu(references, hypothesis):
    ref_tokens = [list(ref) for ref in references]
    hyp_tokens = list(hypothesis)
    return sentence_bleu(ref_tokens, hyp_tokens)

# 4️⃣ 计算 BLEU 分数
references = ["明月几时有 把酒问青天", "大江东去 浪淘尽千古风流人物"]
hypothesis = "明月几时有 何时再相见"

bleu_score = evaluate_bleu(references, hypothesis)
print(f"🌟 BLEU Score: {bleu_score:.4f}")
