import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils import Config
from data import Audio_Dataset, Audio_train, Audio_val
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from datetime import datetime
import math


def Softmax(x):
    under = 0
    for i, value in enumerate(x):
        under += math.exp(value - max(x))
    for i, value in enumerate(x):
        x[i] = math.exp(value - max(x)) / under
    return x

# run timestamp
ts = datetime.now().strftime('%Y%m%d_%H%M%S')

# default `log_dir` is "runs" - we'll be more specific here
train_seq_length = 10
hidden_size = 128
num_layers = 2
bidirectional = True
batch_size = 100
learning_rate = 0.01
num_epochs = 100
offset_random = False

name = 'seq_len={}, hidden={}, layers={}, bid={}, batch={}, lr={}, epoch={}, offset_random={}'.format(
    train_seq_length,
    hidden_size,
    num_layers,
    bidirectional,
    batch_size,
    learning_rate,
    num_epochs,
    offset_random
)
writer = SummaryWriter(f'runs/hw5/{name}')


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bidirectional):
        super(Model, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
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
        return torch.zeros(number * num_layers, batch_size, self.hidden_size)

    def forward(self, x):
        batch_size = x.size(1)
        h = self.init_hidden(batch_size).to(device)
        out, h = self.rnn(x, h)
        out = self.dropout(out)
        out = self.fc(out)
        # return out, h
        return out


dataset = Audio_Dataset(train_seq_length, offset_random)
train_dataset = Audio_train(dataset.X_train, dataset.y_train)
val_dataset = Audio_val(dataset.X_val, dataset.y_val)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


model = Model(input_size=64, hidden_size=hidden_size, output_size=3, bidirectional=bidirectional)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(device)
model.to(device)

summary(
    model,
    input_size=(train_seq_length, 64)
)
# writer.add_graph(model, torch.zeros(batch_size, train_seq_length, 64))

loss_func = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

### TRAIN
best_test_acc = 0.0
best_model_epoch = 0
torch.save(model, f'./runs/hw5/{name}/best_model.pth')
iter_count = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_acc = 0
    i = 0
    for x, y in train_dataloader:
        i += 1
        iter_count += 1
        x = x.to(device)
        y = y.to(device)
        yhat = model(x)
        loss = loss_func(yhat, y)
        # loss = loss_func(yhat[:, 5:, :], y[:, 5:, :])
        model.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss
        # acc part
        for item in range(0, batch_size):
            predict_in_length = torch.argmax(yhat[item], axis=1)
            count = np.bincount(predict_in_length.cpu().numpy())
            predict = np.argmax(count)
            if predict == torch.argmax(y[item], axis=1)[0]:
                train_acc += 1
        if (i > 30) and not (i % 10):
            writer.add_scalar(f'Accuracy/train/',
                              train_acc / i / batch_size,
                              iter_count)
            writer.add_scalar(f'Loss/train/',
                              train_loss / i / batch_size,
                              iter_count)
    if not (epoch % 1):
        print(f'Epoch: {epoch + 1:02d}, ' +
              f'Loss: {train_loss / len(train_dataloader.dataset):.4f}, ' +
              f'Acc: {train_acc / len(train_dataloader.dataset):.4f}')

    # TEST
    model.eval()
    test_loss = 0
    test_acc = 0
    j = 0
    for x, y in val_dataloader:

        j += 1
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            yhat = model(x)
        loss = loss_func(yhat, y)
        test_loss += loss
        # acc part
        for item in range(0, batch_size):
            predict_in_length = torch.argmax(yhat[item], axis=1)
            count = np.bincount(predict_in_length.cpu().numpy())
            predict = np.argmax(count)
            if predict == torch.argmax(y[item], axis=1)[0]:
                test_acc += 1

        iter_test_count = 0
        softmax_saver = {
            'y': [],
            'x': [],
            'language': [],
        }
        for item in range(0, batch_size):
            iter_test_count += 1
            sum_value = [0, 0, 0]
            for it, value in enumerate(yhat[item]):
                sum_value += Softmax(value.cpu().numpy())
            sum_value /= train_seq_length
            softmax_saver['y'].append(sum_value[0])
            softmax_saver['language'].append('english')
            softmax_saver['x'].append(iter_test_count)
            softmax_saver['y'].append(sum_value[1])
            softmax_saver['language'].append('hindi')
            softmax_saver['x'].append(iter_test_count)
            softmax_saver['y'].append(sum_value[2])
            softmax_saver['language'].append('mandarin')
            softmax_saver['x'].append(iter_test_count)


        # if (j > 30) and not (j % 10):
        writer.add_scalar(f'Accuracy/test/',
                          test_acc / j / batch_size,
                          iter_count)

        writer.add_scalar(f'Loss/test/',
                          test_loss / j / batch_size,
                          iter_count)
    # plot 3 softmax
    import seaborn as sns
    sns.set(style="ticks", palette="pastel")
    import pandas as pd
    acc = softmax_saver
    acc = pd.DataFrame(acc)
    import matplotlib.pyplot as plt
    plt.figure()
    # plt.plot(softmax_saver['x'], softmax_saver['0'], name=)
    sns.lineplot(x="x", y="y", data=acc, hue='language')
    plt.title(f'ep={epoch} in val')
    plt.xlabel('time')
    plt.ylabel('prob.')
    plt.show()

    # Save best model
    this_test_acc = test_acc / len(val_dataloader.dataset)
    if this_test_acc > best_test_acc:
        best_test_acc = this_test_acc
        best_model_epoch = epoch
        torch.save(model, f'./runs/hw5/{name}/best_model.pth')

    print(f'Test Loss:{test_loss / len(val_dataloader.dataset):.4f}, ' +
              f'Test Acc: {test_acc / len(val_dataloader.dataset):.4f}, ' +
          f'Best_model_epoch:{best_model_epoch}')

print('Finished Training')
