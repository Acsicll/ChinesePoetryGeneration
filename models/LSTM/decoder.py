import torch
import torch.nn as nn
from sympy.physics.units.systems.si import dimex


class Decoder_LSTM(nn.Module):
    def __init__(self, output_dim,emb_dim, hidden_dim, num_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(hidden_dim * 2 + emb_dim, hidden_dim, num_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim * 3 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.num_directions = 2 if self.rnn.bidirectional else 1

        self.attn_combine = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.hidden_transform = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0)  # [1, batch_size]
        embedded = self.dropout(self.embedding(input))  # [1, batch_size, emb_dim]

        attention_weights = self.attention(hidden, encoder_outputs)  # [batch_size, seq_len]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [batch_size, seq_len, hidden_dim * 2]
        attention_weights = attention_weights.unsqueeze(1)  # [batch_size, 1, seq_len]
        attention_applied = torch.bmm(attention_weights, encoder_outputs)  # [batch_size, 1, hidden_dim * 2]
        attention_applied = attention_applied.squeeze(1)  # [batch_size, hidden_dim * 2]

        rnn_input = torch.cat((embedded.squeeze(0),attention_applied), dim=1)  # [batch_size, hidden_dim * 2 + emb_dim]
        rnn_input = rnn_input.unsqueeze(0)  # [1, batch_size, hidden_dim * 2 + emb_dim]
        hidden = hidden.contiguous()
        cell = cell.contiguous()

        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))  # output: [1, batch_size, hidden_dim]
        output = torch.cat((output.squeeze(0), attention_applied, embedded.squeeze(0)),
                           dim=1)  # [batch_size, hidden_dim * 3 + emb_dim]
        prediction = self.fc_out(output)  # [batch_size, output_dim]
        return prediction, hidden, cell
