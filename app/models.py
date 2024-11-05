import torch
import torch.nn as nn


class ForexLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ForexLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class ForexCNNLSTM(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 num_classes,
                 cnn_channels=32,
                 kernel_size=3):
        super(ForexCNNLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # CNN layer
        self.cnn = nn.Conv1d(in_channels=input_size,
                             out_channels=cnn_channels,
                             kernel_size=kernel_size,
                             padding=kernel_size // 2)
        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(cnn_channels,
                            hidden_size,
                            num_layers,
                            batch_first=True,
                            bidirectional=True)
        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)
        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.cnn(x))
        x = x.permute(0, 2, 1)
        # Initialize LSTM hidden state
        h0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0),
                         self.hidden_size).to(x.device)
        # LSTM layer
        out, _ = self.lstm(x, (h0, c0))
        # Attention mechanism
        attn_weights = torch.softmax(self.attention(out), dim=1)
        out = torch.sum(out * attn_weights, dim=1)
        # Dropout layer
        out = self.dropout(out)
        # Fully connected layer
        out = self.fc(out)
        return out
