import sys
import os
import random

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


class GunSoundDataset(Dataset):
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


if __name__ == '__main__':
    dirpath = './data/gun_sound_v1'

    exp1_train_dataset = GunSoundDataset(dirpath, './data/exp1_train.csv', 44100, 3, 'train')
    exp1_test_dataset = GunSoundDataset(dirpath, './data/exp1_test.csv', 44100, 3, 'test')
    exp2_train_dataset = GunSoundDataset(dirpath, './data/exp2_train.csv', 44100, 3, 'train')
    exp2_test_dataset = GunSoundDataset(dirpath, './data/exp2_test.csv', 44100, 3, 'test')
    exp3_train_dataset = GunSoundDataset(dirpath, './data/exp3_train.csv', 44100, 3, 'train')
    exp3_test_dataset = GunSoundDataset(dirpath, './data/exp3_test.csv', 44100, 3, 'test')

    wv, cate, dist, dire = exp1_train_dataset[0]
    print(len(exp1_train_dataset), len(exp1_test_dataset), wv.shape, cate, dist, dire)
    wv, cate, dist, dire = exp2_train_dataset[0]
    print(len(exp2_train_dataset), len(exp2_test_dataset), wv.shape, cate, dist, dire)
    wv, cate, dist, dire = exp3_train_dataset[0]
    print(len(exp3_train_dataset), len(exp3_test_dataset), wv.shape, cate, dist, dire)


