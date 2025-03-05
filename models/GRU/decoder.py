import torch
import torch.nn as nn

class Decoder_GRU(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.rnn = nn.GRU(hidden_dim * 2 + emb_dim, hidden_dim, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 3 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0) # [1, batch_size]
        embedded = self.dropout(self.embedding(input)) # [1, batch_size, emb_dim]

        attention_weights = self.attention(hidden, encoder_outputs) # [batch_size, seq_len]
        encoder_outputs = encoder_outputs.permute(1, 0, 2) # [batch_size, seq_len, hidden_dim]
        attention_weights = attention_weights.unsqueeze(1) # [batch_size, 1, seq_len]
        attention_applied = torch.bmm(attention_weights, encoder_outputs) # [batch_size, 1, hidden_dim *2]
        attention_applied = attention_applied.squeeze(1) # [batch_size, hidden_dim *2]

        rnn_input = torch.cat((embedded.squeeze(0), attention_applied), dim=1) # [batch_size, hidden_dim * 2 + emb_dim]
        rnn_input = rnn_input.unsqeeze(0) # [1, batch_size, hidden_dim * 2 + emb_dim]

        output, hidden = self.rnn(rnn_input, hidden)
        output = torch.cat((outptut.unsqueeze(0), attention_applied, embedded.squeeze(0)), dim=1)
        prediction = self.fc(output)

        return prediction, hidden