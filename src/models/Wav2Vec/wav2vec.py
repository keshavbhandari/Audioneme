import torch
from torch import nn

class Wav2Vec_Classification(nn.Module):
    def __init__(self, wav2vec_model, vocab_size):
        super().__init__()

        self.encoder = wav2vec_model

        self.reducer = nn.Sequential(
            nn.Linear(256, 64, bias = True),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1, bias = True)
        )

        self.audio_lstm = nn.LSTM(input_size=768, hidden_size=128, num_layers=4, batch_first = True, bidirectional = False)
        self.transcription_lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first = True, bidirectional = False)
        self.embedding = nn.Embedding(vocab_size, 128)

    def forward(self, x, t):
        x = x.squeeze(1)
        x, _ = self.encoder.extract_features(x)
        x = x[-1]
        x, _ = self.audio_lstm(x)
        x = torch.sum(x, dim=1)
        t = self.embedding(t)
        t = torch.sum(t, dim=1)
        x = torch.cat((x, t), dim=1)
        x = self.reducer(x)
        return x