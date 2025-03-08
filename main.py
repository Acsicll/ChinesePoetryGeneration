import torch
from torch import optim, nn

from SongIambicsGeneration.dataset import load_vocab, get_dataloader
from SongIambicsGeneration.generate import generate_poetry, generate_acrostic_poetry
from SongIambicsGeneration.models.attention import Attention
from SongIambicsGeneration.models.LSTM.decoder import Decoder_LSTM
from SongIambicsGeneration.models.LSTM.encoder import Encoder_LSTM
from SongIambicsGeneration.models.LSTM.seq2seq import Seq2Seq_LSTM
from SongIambicsGeneration.models.GRU.decoder import Decoder_GRU
from SongIambicsGeneration.models.GRU.encoder import Encoder_GRU
from SongIambicsGeneration.models.GRU.seq2seq import Seq2Seq_GRU
from SongIambicsGeneration.preprocess import preprocess_data
from SongIambicsGeneration.train import to_train
from my_config import HyperParams

"""
GRU - 0-3
"""


def init_lstm_model(hparams, attn, device,pretrained_embedding):
    enc = Encoder_LSTM(hparams.input_dim, hparams.enc_emb_dim, hparams.hidden_dim, hparams.num_layers,
                       hparams.enc_dropout,pretrained_embedding).to(device)
    dec = Decoder_LSTM(hparams.output_dim, hparams.dec_emb_dim, hparams.hidden_dim, hparams.num_layers,
                       hparams.dec_dropout,
                       attn,pretrained_embedding).to(device)
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

    tokenizer, pretrained_embedding = preprocess_data()
    train_loader, val_loader = get_dataloader(batch_size=hparams.batch_size)
    # tokenizer = load_vocab()

    hparams.input_dim = hparams.output_dim = len(tokenizer[0])

    attn = Attention(hparams.hidden_dim).to(device)
    hparams.model_type = "LSTM"
    model = init_lstm_model(hparams, attn, device,pretrained_embedding)
    #model = init_gru_model(hparams, attn, device)

    optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.98), eps=1e-09)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding token

    epoch = [40, 50, 1]
    to_train(model_type = hparams.model_type,
             epoch = epoch,
             model = model,
             train_loader = train_loader,
             val_loader = val_loader,
             learning_rate =hparams.learning_rate,
             batch_size = hparams.batch_size,
             optimizer = optimizer,
             criterion = criterion,
             clip = hparams.clip,
             device = device)
    # keywords = ['春风', '烟雨']
    # words = "江南烟雨"
    # epoch_range = [30, 40, 1]
    # generate_poetry(model_type = hparams.model_type,
    #             epoch_range = epoch_range,
    #             timestamp = "20250308",
    #             model = model,
    #             tokenizer = tokenizer,
    #             device = device,
    #             keywords = keywords,
    #             temperature = 0.7,
    #             beam_size = 2,
    #             top_p=0.9)

    """
    learning_rate/ clip/ hidden_dim / temperature / beam_size / top_p 调参
    """

