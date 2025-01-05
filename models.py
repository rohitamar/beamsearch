import math 
import torch 
import torch.nn as nn

class NPLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, block_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.hidden = nn.Linear((block_size - 1) * embed_size, hidden_size)
        self.final = nn.Linear(hidden_size, vocab_size)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.embed(input)
        x = x.reshape(x.shape[0], -1)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.final(x)
        return x

class RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.final = nn.Linear(hidden_size, vocab_size)

    def forward(self, input):
        input = self.embed(input)
        lstm_out, _ = self.lstm(input)
        out = self.final(lstm_out)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len: int = 1000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = PositionalEncoding(d_model=d_model)

        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True), num_layers=4)
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1])

        x = self.embedding(x)
        x = self.position(x)
        x = self.encoder(x, mask)
        x = self.decoder(x)
        x = torch.log_softmax(x, dim = -1)
        
        return x