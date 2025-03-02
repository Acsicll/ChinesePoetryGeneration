import torch
from torch import optim, nn

from SongIambicsGeneration.dataset import load_sentences, load_vocab, get_dataloader
from SongIambicsGeneration.generate import generate_poetry_with_split_keywords, \
    generate_poetry_v1, generate_poetry_v2
from SongIambicsGeneration.models.attention import Attention
from SongIambicsGeneration.models.decoder import Decoder
from SongIambicsGeneration.models.encoder import Encoder
from SongIambicsGeneration.models.seq2seq import Seq2Seq
from SongIambicsGeneration.train import to_train

INPUT_DIM = 5000
OUTPUT_DIM = 5000
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HIDDEN_DIM = 1024
NUM_LAYERS = 2
ENC_DROUPUT = 0.5
DEC_DROUPUT = 0.5
BATCH_SIZE = 64
CLIP = 1

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    train_loader = get_dataloader(batch_size=BATCH_SIZE)
    word_2_idx, idx_2_word = load_vocab()

    INPUT_DIM = len(word_2_idx)
    OUTPUT_DIM = len(word_2_idx)

    sentences = load_sentences()
    #embedding = get_Word2vec(word_2_idx,sentences,ENC_EMB_DIM, device)

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HIDDEN_DIM, NUM_LAYERS, ENC_DROUPUT).to(device)
    attn = Attention(HIDDEN_DIM).to(device)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HIDDEN_DIM, NUM_LAYERS, DEC_DROUPUT, attn).to(device)
    model = Seq2Seq(enc, dec, device).to(device)

    optimizer = optim.AdamW(model.parameters(),lr=5e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # ignore padding token

    #to_train(model, train_loader, optimizer, criterion, CLIP, device)
    keywords = ['江南', '烟雨']
    i = 10
    while i < 20:
        model.load_state_dict(torch.load(f'./SavedModels/seq2seq_{i+1}.pt', map_location=device))
        model.to(device)
        #poem = generate_poem_temperature(model, (word_2_idx, idx_2_word), device, temperature=0.8,max_length=32)
        poem = generate_poetry_v2(model, (word_2_idx, idx_2_word), device, "".join(keywords),total_line=5)
        #poem =  generate_poetry_with_split_keywords(model, (word_2_idx, idx_2_word), device, keywords,temperature=1.2)
        print(f"NO.{i} Generated poem: {poem}")
        i += 1