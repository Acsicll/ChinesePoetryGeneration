import torch
from torch import optim, nn

from SongIambicsGeneration.dataset import load_sentences, load_vocab, get_dataloader
from SongIambicsGeneration.generate import to_generate
from SongIambicsGeneration.models.attention import Attention
from SongIambicsGeneration.models.decoder import Decoder
from SongIambicsGeneration.models.encoder import Encoder
from SongIambicsGeneration.models.seq2seq import Seq2Seq
from SongIambicsGeneration.train import to_train
from my_config import HyperParams

"""

"""
if __name__ == "__main__":
    hparams = HyperParams()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    train_loader = get_dataloader(batch_size=hparams.batch_size)
    tokenizer = load_vocab()

    hparams.input_dim = hparams.output_dim = len(tokenizer[0])

    #sentences = load_sentences()

    enc = Encoder(hparams.input_dim, hparams.enc_emb_dim, hparams.hidden_dim, hparams.num_layers,
                  hparams.enc_dropout).to(device)
    attn = Attention(hparams.hidden_dim).to(device)
    dec = Decoder(hparams.output_dim, hparams.dec_emb_dim, hparams.hidden_dim, hparams.num_layers, hparams.dec_dropout,
                  attn).to(device)
    model = Seq2Seq(enc, dec, device).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding token

    epoch = [0, 10]
    to_train(epoch, model, train_loader, optimizer, criterion, hparams.clip, device)
    keywords = ['江南', '烟雨']
    epoch_range = [0, 10]
    to_generate(epoch_range, model, tokenizer, device, keywords, device)
