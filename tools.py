import torch
from utils import Config
import torch.nn as nn
from streaming_model_class import Model


def gen_autolab_model():

    model = Model()
    model.load_state_dict(torch.load('../log_history/ResNet34-3/best_model.pth', map_location=torch.device('cpu')))

    torch.save(model, '../log_history/ResNet34-3/categorical-model.pth')

    print(model)
    print('Success!')

def plot_model():
    from torchviz import make_dot, make_dot_from_trace
    model = Model()
    x = torch.randn(1, 10, 64)
    y = model(x)
    plot = make_dot(y.mean(), params=dict(model.named_parameters()))
    plot.view()

def learning_curve():
    import seaborn as sns
    sns.set(style="ticks", palette="pastel")
    import pandas as pd

    acc = {'phase': ['train', 'test', 'train', 'test', 'train', 'test', 'train', 'test', 'train', 'test', 'train', 'test', 'train', 'test', 'train', 'test', 'train', 'test', 'train', 'test', 'train', 'test', 'train', 'test', 'train', 'test', 'train', 'test', 'train', 'test', 'train', 'test', 'train', 'test', 'train', 'test', 'train', 'test', 'train', 'test', 'train', 'test', 'train', 'test', 'train', 'test', 'train', 'test', 'train', 'test', 'train', 'test', 'train', 'test', 'train', 'test', 'train', 'test', 'train', 'test'], 'epoch': [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29], 'loss': [6.5168, 0.7101, 1.9671, 0.7083, 0.6945, 0.6948, 1.3306, 0.6976, 0.6953, 0.6932, 1.7724, 0.6934, 0.6913, 0.6942, 1.0375, 0.6932, 0.6932, 0.6924, 0.6916, 0.6976, 0.7008, 0.6914, 0.7484, 0.6936, 0.6927, 0.6939, 0.6898, 0.687, 0.6906, 0.6882, 0.689, 0.7031, 0.6866, 0.6861, 0.6864, 0.6858, 0.685, 0.6834, 0.6835, 0.6839, 0.6831, 0.6863, 0.6814, 0.6809, 0.6796, 0.679, 0.6755, 0.7095, 0.6723, 0.6706, 0.6695, 0.6713, 0.6682, 0.6694, 0.6677, 0.6691, 0.6661, 0.6671, 0.6649, 0.6659], 'acc': [0.5005, 0.4978, 0.5074, 0.5006, 0.5105, 0.5198, 0.5071, 0.4979, 0.5098, 0.5001, 0.5117, 0.4984, 0.5293, 0.5049, 0.5165, 0.494, 0.5132, 0.5153, 0.5291, 0.5378, 0.5239, 0.5312, 0.5332, 0.5103, 0.5153, 0.5014, 0.5424, 0.5478, 0.5437, 0.542, 0.5483, 0.5298, 0.5525, 0.5509, 0.5544, 0.5431, 0.5572, 0.5538, 0.5596, 0.5554, 0.5627, 0.5408, 0.5644, 0.5655, 0.5701, 0.5756, 0.5802, 0.5825, 0.5894, 0.5904, 0.591, 0.5918, 0.5946, 0.5916, 0.5978, 0.5947, 0.5984, 0.5941, 0.5995, 0.598]}


    acc = pd.DataFrame(acc)
    import matplotlib.pyplot as plt

    fig1, f1_axes = plt.subplots(ncols=2, nrows=1, figsize=(10,4))

    sns.lineplot(x="epoch", y="acc", hue="phase", data=acc, ax=f1_axes[0])
    sns.lineplot(x="epoch", y="loss", hue="phase", data=acc, ax=f1_axes[1])
    f1_axes[0].set_xlabel("epoch", size=15)
    f1_axes[1].set_xlabel("epoch", size=15)
    f1_axes[0].set_ylabel("acc", size=15)
    f1_axes[1].set_ylabel("loss", size=15)
    # f1_axes[0].set_xticks(range(0, 30))
    # f1_axes[1].set_xticks(range(0, 30))
    f1_axes[0].set_title("Pairwise-2", size=15)
    f1_axes[1].set_title("Pairwise-2", size=15)

    plt.show()

def summary_model():
    from torchsummary import summary
    device = torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')
    model = Model()
    model.to(device)
    summary(model, input_size=(3, 64))

# learning_curve()
# gen_autolab_model()
# plot_model()
summary_model()
