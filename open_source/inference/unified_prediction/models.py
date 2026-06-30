"""LSTM model definitions for cyclist intention prediction.

Three Bi-LSTM models:
  - ExplicitLSTMClassifier: explicit gesture classification (input=66, layers=4, classes=3)
  - ImplicitLSTMClassifier: implicit gesture classification (input=66, layers=4, classes=4)
  - ManeuverLSTMClassifier: maneuver prediction (input=2, layers=2, classes=3)

Fixed from originals: h0/c0 use x.device instead of a global device variable.
"""

import torch
import torch.nn as nn


class ExplicitLSTMClassifier(nn.Module):
    def __init__(self, input_size=66, hidden_size=128, num_layers=4,
                 num_classes=3, dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0),
                          self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0),
                          self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


class ImplicitLSTMClassifier(nn.Module):
    def __init__(self, input_size=66, hidden_size=128, num_layers=4,
                 num_classes=4, dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0),
                          self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0),
                          self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


class ManeuverLSTMClassifier(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=2,
                 num_classes=3, dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0),
                          self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0),
                          self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out
