from torch import nn

class ResNetSE34V2_Classification(nn.Module):
    def __init__(self, pretrained_model):
        '''config[-1] will be the output sequence length'''
        super().__init__()

        self.encoder = pretrained_model

        # self.lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=False)

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x