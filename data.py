from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from utils import Config
import numpy as np
import os
import time
import random
import audioread
import librosa

class Audio_Dataset:
    def __init__(self, seq_length, offset_random):
        self.dir = Config['data_dir']
        self.seq_length = seq_length
        X_english, y_englist = self.get_MFCC_features_labels('english', offset_random, N_files=40, duration=self.seq_length)
        X_hindi, y_hindi = self.get_MFCC_features_labels('hindi', offset_random, N_files=40, duration=self.seq_length)
        X_mandarin, y_mandarin = self.get_MFCC_features_labels('mandarin', offset_random, N_files=40, duration=self.seq_length)
        self.num_seq = X_english.shape[0] // seq_length
        self.X_english = X_english.reshape((self.num_seq, seq_length, 64))
        self.y_english = y_englist.reshape((self.num_seq, seq_length, 1))
        print('english features shape:{}, labels shape:{}'.format(self.X_english.shape, self.y_english.shape))
        self.X_hindi = X_hindi.reshape((self.num_seq, seq_length, 64))
        self.y_hindi = y_hindi.reshape((self.num_seq, seq_length, 1))
        print('hindi features shape:{}, labels shape:{}'.format(self.X_hindi.shape, self.y_hindi.shape))
        self.X_mandarin = X_mandarin.reshape((self.num_seq, seq_length, 64))
        self.y_mandarin = y_mandarin.reshape((self.num_seq, seq_length, 1))
        print('mandarin features shape:{}, labels shape:{}'.format(self.X_mandarin.shape, self.y_mandarin.shape))
        # concatenate
        X = np.concatenate((self.X_english, self.X_hindi, self.X_mandarin), axis=0)
        y = np.concatenate((self.y_english, self.y_hindi, self.y_mandarin), axis=0)
        print('all features shape:{}, labels shape:{}'.format(X.shape, y.shape))
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.2)


    def get_MFCC_features_labels(self, language, offset_random, N_files, duration):
        all_features_for_this_language = None
        for i in range(1, N_files+1):
            file_dir = os.path.join(self.dir, 'train_'+language, '{}_{:04d}.wav'.format(language, i))
            if offset_random:
                offset = random.randint(0, 600 - duration)
            else:
                offset = 100
            y, sr = librosa.load(file_dir, sr=16000, offset=offset, duration=duration)
            mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft=int(sr*0.025), hop_length=int(sr*0.010))
            mat = mat[:, :-1].T
            if all_features_for_this_language is None:
                all_features_for_this_language = mat
            else:
                all_features_for_this_language = np.concatenate((all_features_for_this_language, mat), axis=0)
        labels = {'english': 0, 'hindi': 1, 'mandarin': 2}
        number_features = all_features_for_this_language.shape[0]
        all_labels_for_this_language = np.ones((number_features, 1)) * labels[language]
        return all_features_for_this_language, all_labels_for_this_language

class Audio_train(Dataset):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    def __len__(self):
        return len(self.X_train)
    def __getitem__(self, index):
        x = self.X_train[index:index + 1]
        x = x.squeeze(0)
        y = self.y_train[index]
        y_onehot = []
        for i in range(0, len(y)):
            if y[i][0] == 0:
                y_onehot.append([1, 0, 0])
            elif y[i][0] == 1:
                y_onehot.append([0, 1, 0])
            else:
                y_onehot.append([0, 0, 1])
        return torch.Tensor(x), torch.Tensor(y_onehot)

class Audio_val(Dataset):
    def __init__(self, X_val, y_val):
        self.X_val = X_val
        self.y_val = y_val
    def __len__(self):
        return len(self.X_val)
    def __getitem__(self, index):
        x = self.X_val[index:index + 1]
        x = x.squeeze(0)
        y = self.y_val[index]
        y_onehot = []
        for i in range(0, len(y)):
            if y[i][0] == 0:
                y_onehot.append([1, 0, 0])
            elif y[i][0] == 1:
                y_onehot.append([0, 1, 0])
            else:
                y_onehot.append([0, 0, 1])
        return torch.Tensor(x), torch.Tensor(y_onehot)
