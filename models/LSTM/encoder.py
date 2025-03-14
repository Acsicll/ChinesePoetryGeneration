import torch
import torch.nn as nn

class Encoder_LSTM(nn.Module):
    def __init__(self, input_dim,emb_dim, hidden_dim,num_layers,droupout,pretrain_emb):
        super().__init__()
        #self.embedding = nn.Embedding(input_dim, emb_dim)
        self.embedding = pretrain_emb
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers, dropout=droupout,bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, hidden_dim) # bidirectional LSTM has 2*hidden_dim output
        self.dropout = nn.Dropout(droupout)
        self.num_directions = 2 if self.rnn.bidirectional else 1
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, src):
        # src shape [batch_size, seq_len]
        embedded = self.dropout(self.embedding(src))  # [batch_size, seq_len, emb_dim]
        embedded = embedded.permute(1, 0, 2)  # [seq_len, batch_size, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)  # outputs: [seq_len, batch_size, hidden_dim * 2]

        # 处理 hidden
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)  # [num_layers, 2, batch_size, hidden_dim]
        hidden = hidden[:, -1, :, :]  # 取最后一个方向的隐藏状态，形状为 [num_layers, batch_size, hidden_dim]

        # 处理 cell
        cell = cell.view(self.num_layers, 2, -1, self.hidden_dim)  # [num_layers, 2, batch_size, hidden_dim]
        cell = cell[:, -1, :, :]  # 取最后一个方向的细胞状态，形状为 [num_layers, batch_size, hidden_dim]

        return outputs, hidden, cell
