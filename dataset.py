import sys
import os
import random
import librosa

import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

from torch.utils.data import Dataset

import pandas as pd
import numpy as np


class RandomClip:
    def __init__(self, sample_rate, clip_length):
        self.clip_length = clip_length

    def __call__(self, audio_data):
        audio_length = audio_data.shape[1] # (chanel, time)

        if audio_length > self.clip_length:
            offset = random.randint(0, audio_length-self.clip_length)
            audio_data = audio_data[:, offset:(offset+self.clip_length)]

        return audio_data


class CenterCrop:
    def __init__(self, sample_rate, clip_length):
        self.clip_length = clip_length

    def __call__(self, audio_data):
        audio_length = audio_data.shape[1] # (chanel, time)

        if audio_length > self.clip_length:
            offset = audio_length //2 - self.clip_length//2
            audio_data = audio_data[:, offset:(offset+self.clip_length)]

        return audio_data


class BGGunDataset(Dataset):
    def __init__(self, dirpath, path, sample_rate, input_sec, phase, dicts=None):
        self.dirpath = dirpath
        self.phase = phase
        self.df = pd.read_csv(path)
        self.sr = sample_rate
        self.input_sec = input_sec
        self.dicts = dicts

        self.rclip = RandomClip(self.sr, self.sr*self.input_sec)
        self.ccrop = CenterCrop(self.sr, self.sr*self.input_sec)

        if dicts is None:
            dicts = dict()
            class_dict = {cat:idx for idx, cat in enumerate(self.df['cate'].value_counts().index.tolist())}
            dist_dict = {cat:idx for idx, cat in enumerate(self.df['dist'].value_counts().index.tolist())}
            dire_dict = {cat:idx for idx, cat in enumerate(self.df['dire'].value_counts().index.tolist())}
            dicts['cate'] = class_dict
            dicts['dist'] = dist_dict
            dicts['dire'] = dire_dict
            self.dicts = dicts

    def __getitem__(self, index):
        inst = self.df.iloc[index]
        cate = self.dicts['cate'][inst['cate']]
        dist = self.dicts['dist'][inst['dist']]
        dire = self.dicts['dire'][inst['dire']]

        waveform, sr = torchaudio.load(os.path.join(self.dirpath, inst['name']))
        assert sr == self.sr
        if self.phase == 'train':
            cropped_wv = self.rclip(waveform)
        else:
            cropped_wv = self.ccrop(waveform)
        #print(inst['path'], cropped_wv.shape)
        return cropped_wv, torch.tensor(cate).long(), torch.tensor(dist).long(), torch.tensor(dire).long()

    def __len__(self):
        return len(self.df)


class ForenGunDataset(Dataset):
    def __init__(self, dirpath, path, sample_rate, input_sec, phase, n_train=None, dicts=None):
        self.dirpath = dirpath
        self.phase = phase
        self.n_train = n_train
        self.df = pd.read_csv(path)
        self.sr = sample_rate
        self.input_sec = input_sec
        self.dicts = dicts

        self.rclip = RandomClip(self.sr, self.sr*self.input_sec)
        self.ccrop = CenterCrop(self.sr, self.sr*self.input_sec)

        if dicts is None:
            dicts = dict()
            class_dict = {cat:idx for idx, cat in enumerate(self.df['cate'].value_counts().index.tolist())}
            dicts['cate'] = class_dict

            self.dicts = dicts

        self.filterbyfold()

    def filterbyfold(self):
        if self.phase == 'train':
            if self.n_train:
                self.df = self.df.iloc[:self.n_train]
        elif self.phase == 'val':
            pass
        elif self.phase == 'test':
            pass
        else:
            raise Exception(f'Not Supported mode:{self.phase}')

    def __getitem__(self, index):
        inst = self.df.iloc[index]
        label = self.dicts['cate'][inst['cate']]

        # waveform, sr = torchaudio.load(inst['path'])
        tokens = inst['path'].split('/')
        filename = tokens[-1].replace('.wav', '.npy')
        folder = tokens[-2]
        # waveform, sr = librosa.load(os.path.join(self.dirpath, inst['path']), mono=False, sr=96000)
        # waveform = torch.from_numpy(librosa.resample(waveform, sr, self.sr))
        waveform = torch.from_numpy(np.load(os.path.join(self.dirpath, 'GunshotAudioForensicsDataset/numpy', f'{folder}@{filename}')))
        sr = self.sr
        _, L = waveform.shape
        length = self.sr*self.input_sec if L < self.sr*self.input_sec else L
        tmp = torch.zeros((2, length))
        tmp[:, :L] = waveform
        waveform = tmp
        assert sr == self.sr
        if self.phase == 'train':
            cropped_wv = self.rclip(waveform)
        else:
            cropped_wv = self.ccrop(waveform)
        #         print(inst['path'], cropped_wv.shape)
        return cropped_wv, torch.tensor(label).long(), torch.zeros(1), torch.zeros(1)

    def __len__(self):
        return len(self.df)


class UrbanGunDataset(Dataset):
    def __init__(self, dirpath, path, sample_rate, input_sec, phase, folds, use_bgg=False, dicts=None):
        self.dirpath = dirpath
        self.phase = phase
        self.use_bgg = use_bgg
        self.df = pd.read_csv(path)
        self.sr = sample_rate
        self.input_sec = input_sec
        self.dicts = dicts
        self.folds = folds

        self.rclip = RandomClip(self.sr, self.sr*self.input_sec)
        self.ccrop = CenterCrop(self.sr, self.sr*self.input_sec)

        if dicts is None:
            dicts = dict()
            class_dict = {cat:idx for idx, cat in enumerate(self.df['class'].value_counts().index.tolist())}
            dicts['cate'] = class_dict

            self.dicts = dicts

        self.filterbyfold()

    def filterbyfold(self):
        folds = self.folds
        if self.phase == 'train':
            if self.use_bgg:
                folds.append(999)
            self.df = self.df[self.df['fold'].isin(folds)]
        elif self.phase == 'val':
            self.df = self.df[self.df['fold'].isin(folds)]
        elif self.phase == 'test':
            self.df = self.df[self.df['fold'].isin(folds)]
        else:
            raise Exception(f'Not Supported mode:{self.phase}')

    def __getitem__(self, index):
        inst = self.df.iloc[index]
        label = 1 if inst['class'] == 'gun_shot' else 0
        # label = self.dicts['cate'][inst['class']]
        if inst['fold'] == 999:
            path = os.path.join('./data/gun_sound_v1', inst['slice_file_name'])
            waveform, sr = torchaudio.load(path)

        else:
            path = os.path.join(self.dirpath, f"fold{inst['fold']}", f"{inst['slice_file_name'].split('.')[0]}.npy")
            waveform = torch.from_numpy(np.load(path))

        sr = self.sr
        if len(waveform.shape) == 1:
            waveform = torch.stack([waveform, waveform], dim=0)
        _, L = waveform.shape
        length = self.sr*self.input_sec if L < self.sr*self.input_sec else L
        tmp = torch.zeros((2, length))
        tmp[:, :L] = waveform
        waveform = tmp
        assert sr == self.sr
        if self.phase == 'train':
            cropped_wv = self.rclip(waveform)
        else:
            cropped_wv = self.ccrop(waveform)
        #         print(inst['path'], cropped_wv.shape)
        return cropped_wv, torch.tensor(label).long(), torch.zeros(1), torch.zeros(1)

    def __len__(self):
        return len(self.df)




if __name__ == '__main__':
    dirpath = './data/gun_sound_v1'

    exp1_train_dataset = BGGunDataset(dirpath, './data/exp1_train.csv', 44100, 3, 'train')
    exp1_test_dataset = BGGunDataset(dirpath, './data/exp1_test.csv', 44100, 3, 'test')
    exp2_train_dataset = BGGunDataset(dirpath, './data/exp2_train.csv', 44100, 3, 'train')
    exp2_test_dataset = BGGunDataset(dirpath, './data/exp2_test.csv', 44100, 3, 'test')
    exp3_train_dataset = BGGunDataset(dirpath, './data/exp3_train.csv', 44100, 3, 'train')
    exp3_test_dataset = BGGunDataset(dirpath, './data/exp3_test.csv', 44100, 3, 'test')

    wv, cate, dist, dire = exp1_train_dataset[0]
    print(len(exp1_train_dataset), len(exp1_test_dataset), wv.shape, cate, dist, dire)
    wv, cate, dist, dire = exp2_train_dataset[0]
    print(len(exp2_train_dataset), len(exp2_test_dataset), wv.shape, cate, dist, dire)
    wv, cate, dist, dire = exp3_train_dataset[0]
    print(len(exp3_train_dataset), len(exp3_test_dataset), wv.shape, cate, dist, dire)


