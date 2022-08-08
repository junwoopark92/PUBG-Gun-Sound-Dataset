import glob
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
import shutil
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import DataLoader

import argparse
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import time
from IPython.display import Audio, display
from sklearn.metrics import f1_score, accuracy_score

import numpy as np

from dataset import BGGunDataset
from models import CNNExtractor, RNNExtractor, CRNNExtractor, TransformerExtractor, CTransExtractor, MultitaskClassifer, \
    OurClassifer, OurExtractor

if __name__ == '__main__':
    print(torch.__version__)
    print(torchaudio.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='CNN', type=str, choices=['CNN', 'RNN', 'CRNN', 'Trans', 'CTrans' ,'Our'])
    parser.add_argument('--datadir', default='./data/gun_sound_v1', type=str)
    parser.add_argument('--datapath', type=str)
    parser.add_argument('--input_sec', default=3, type=int)
    parser.add_argument('--sr', default=44100, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--hdim', default=64, type=int)
    parser.add_argument('--load', default=None, type=str)

    parser.add_argument('--save_midi', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    print(args)

    # load dataset
    train_df = pd.read_csv('./data/v3_exp2_train.csv')
    test_df = pd.read_csv('./data/v3_exp2_test.csv')
    train_dataset = BGGunDataset(args.datadir, train_df, args.sr, args.input_sec, 'train')
    test_dataset = BGGunDataset(args.datadir, test_df, args.sr, args.input_sec, 'test',
                                   dicts=train_dataset.dicts)

    print(train_dataset.dicts)
    demo_wv, sample_rate = torchaudio.load(args.datapath)
    window_size = args.sr*args.input_sec
    stride = args.sr
    splits = [demo_wv[:, i:i+window_size] for i in range(0,demo_wv.size(1)-window_size+1,stride)]
    demo_wv = torch.stack(splits)
    print(demo_wv.shape)

    label_dicts = train_dataset.dicts
    print(f'EXP1 Training:{len(train_dataset)} Test:{len(test_dataset)}')
    print()

    # build model
    if args.backbone == 'CNN':
        backbone = CNNExtractor(args.hdim, sample_rate=args.sr, n_fft=512, n_mels=96).cuda()
    elif args.backbone == 'RNN':
        backbone = RNNExtractor(args.hdim, sample_rate=args.sr, n_fft=512, n_mels=96).cuda()
    elif args.backbone == 'CRNN':
        backbone = CRNNExtractor(args.hdim, sample_rate=args.sr, n_fft=512, n_mels=96).cuda()
    elif args.backbone == 'Trans':
        backbone = TransformerExtractor(args.hdim, sample_rate=args.sr, n_fft=512, n_mels=96).cuda()
    elif args.backbone == 'CTrans':
        backbone = CTransExtractor(args.hdim, sample_rate=args.sr, n_fft=512, n_mels=96).cuda()
    elif args.backbone == 'Our':
        backbone = OurExtractor(args.hdim, sample_rate=args.sr, n_fft=512, n_mels=96).cuda()
    else:
        raise Exception(f'Not Supported Backbone:{args.backbone}')

    if args.backbone != 'Our':
        classifier = MultitaskClassifer(args.hdim, len(label_dicts['cate']),
                                        len(label_dicts['dist']), len(label_dicts['dire'])).cuda()
    elif args.backbone == 'Our':
        classifier = OurClassifer(args.hdim, len(label_dicts['cate']),
                                  len(label_dicts['dist']), len(label_dicts['dire'])).cuda()
    print(backbone)
    print(classifier)
    params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f'# of Backbone Parameters:{params}')

    # load state dict
    backbone.load_state_dict(torch.load(f'demo_{args.backbone}_backbone.pt'))
    classifier.load_state_dict(torch.load(f'demo_{args.backbone}_classifier.pt'))
    backbone.eval()
    classifier.eval()

    with torch.no_grad():
        demo_wv = demo_wv.cuda()

        features = backbone(demo_wv)
        cate_out, dist_out, dire_out = classifier(features)
        cate_out, dist_out, dire_out = cate_out.cpu().numpy(), dist_out.cpu().numpy(), dire_out.cpu().numpy()
        np.save('./cate_out', cate_out)
        np.save('./dist_out', dist_out)
        np.save('./dire_out', dire_out)


