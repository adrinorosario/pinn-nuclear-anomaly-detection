import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):

    def __init__(self, n_features, hidden_size=64):
        super(LSTMAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=n_features,
            batch_first=True
        )

    def forward(self, x):

        # Encode input sequence
        _, (hidden, _) = self.encoder(x)

        # Repeat hidden state across timesteps
        repeated = hidden.repeat(x.size(1), 1, 1).permute(1, 0, 2)

        # Decode sequence
        decoded, _ = self.decoder(repeated)

        return decoded