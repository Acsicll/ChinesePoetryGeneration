import torch
from torch import optim, nn

from SongIambicsGeneration.dataset import load_vocab, get_dataloader
from SongIambicsGeneration.generate import to_generate
from SongIambicsGeneration.models.attention import Attention
from SongIambicsGeneration.models.LSTM.decoder import Decoder_LSTM
from SongIambicsGeneration.models.LSTM.encoder import Encoder_LSTM
from SongIambicsGeneration.models.LSTM.seq2seq import Seq2Seq_LSTM
from SongIambicsGeneration.models.GRU.decoder import Decoder_GRU
from SongIambicsGeneration.models.GRU.encoder import Encoder_GRU
from SongIambicsGeneration.models.GRU.seq2seq import Seq2Seq_GRU
from SongIambicsGeneration.train import to_train
from my_config import HyperParams

"""

"""


def init_lstm_model(hparams, attn, device):
    enc = Encoder_LSTM(hparams.input_dim, hparams.enc_emb_dim, hparams.hidden_dim, hparams.num_layers,
                       hparams.enc_dropout).to(device)
    dec = Decoder_LSTM(hparams.output_dim, hparams.dec_emb_dim, hparams.hidden_dim, hparams.num_layers,
                       hparams.dec_dropout,
                       attn).to(device)
    model = Seq2Seq_LSTM(enc, dec, device).to(device)
    return model


def init_gru_model(hparams, attn, device):
    enc = Encoder_GRU(hparams.input_dim, hparams.enc_emb_dim, hparams.hidden_dim, hparams.num_layers,
                      hparams.enc_dropout).to(device)
    dec = Decoder_GRU(hparams.output_dim, hparams.dec_emb_dim, hparams.hidden_dim, hparams.num_layers,
                      hparams.dec_dropout,
                      attn).to(device)
    model = Seq2Seq_GRU(enc, dec, device).to(device)
    return model


if __name__ == "__main__":
    hparams = HyperParams()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader = get_dataloader(batch_size=hparams.batch_size)
    tokenizer = load_vocab()

    hparams.input_dim = hparams.output_dim = len(tokenizer[0])

    attn = Attention(hparams.hidden_dim).to(device)
    hparams.model_type = "LSTM"
    model = init_lstm_model(hparams, attn, device)

    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding token

    epoch = [0, 10]
    to_train(model_type = hparams.model_type,
             epoch = epoch,
             model = model,
             train_loader = train_loader,
             learning_rate =hparams.learning_rate,
             batch_size = hparams.batch_size,
             optimizer = optimizer,
             criterion = criterion,
             clip = hparams.clip,
             device = device)
    keywords = ['江南', '烟雨']
    epoch_range = [0, 10]
    to_generate(model_type = hparams.model_type,
                epoch_range = epoch_range,
                model = model,
                tokenizer = tokenizer,
                device = device,
                keywords = keywords)
