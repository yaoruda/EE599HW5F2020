import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_size = 64
        self.hidden_size = 128
        self.output_size = 3
        self.bidirectional = True
        self.num_layers = 2

        self.rnn = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0.2,
            batch_first=False,
            bidirectional=self.bidirectional
        )
        if self.bidirectional:
            self.fc = nn.Linear(self.hidden_size * 2, self.output_size)
        else:
            self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(0.2)

    # create function to init state
    def init_hidden(self, batch_size):
        number = 1
        if self.bidirectional:
            number = 2
        return torch.zeros(number * self.num_layers, batch_size, self.hidden_size)

    def forward(self, x):
        batch_size = x.size(1)
        h = self.init_hidden(batch_size).to(self.device)
        out, h = self.rnn(x, h)
        out = self.dropout(out)
        out = self.fc(out)
        # return out, h
        return out

# model = Model()