import glob
import os
os.environ['CUDA_VISIBLE_DEVICES']='4'
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

from dataset import GunSoundDataset
from models import CNNExtractor, RNNExtractor, CRNNExtractor, TransformerExtractor, CTransExtractor, SingleClassifer

if __name__ == '__main__':
    print(torch.__version__)
    print(torchaudio.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='CNN', type=str, choices=['CNN', 'RNN', 'CRNN', 'Trans', 'CTrans'])
    parser.add_argument('--datadir', default='./data/gun_sound_v1', type=str)
    parser.add_argument('--input_sec', default=3, type=int)
    parser.add_argument('--sr', default=44100, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--hdim', default=64, type=int)

    parser.add_argument('--save_midi', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    print(args)

    # load dataset
    train_dataset = GunSoundDataset(args.datadir, './data/exp1_train.csv', args.sr, args.input_sec, 'train')
    test_dataset = GunSoundDataset(args.datadir, './data/exp1_test.csv', args.sr, args.input_sec, 'test',
                                   dicts=train_dataset.dicts)
    label_dicts = train_dataset.dicts
    print(f'EXP1 Training:{len(train_dataset)} Test:{len(test_dataset)}')
    print()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

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
    else:
        raise Exception(f'Not Supported Backbone:{args.backbone}')

    classifier = SingleClassifer(args.hdim, len(label_dicts['cate'])).cuda()
    print(backbone)
    print(classifier)
    params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f'# of Backbone Parameters:{params}')

    ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(list(backbone.parameters()) + list(classifier.parameters()), lr=args.lr)

    for epoch in range(args.epochs):
        backbone.train()
        classifier.train()

        ce_loss = 0.0
        cate_preds, cate_trues = [], []
        for step, (wv, cate, dist, dire) in enumerate(train_loader):
            wv, cate = wv.cuda(), cate.cuda()

            features = backbone(wv)
            cate_out = classifier(features)
            cate_loss = ce(cate_out, cate)
            cate_pred = torch.argmax(cate_out, dim=-1)

            loss = cate_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            ce_loss += loss.item()

            cate_preds.append(cate_pred.detach().cpu())
            cate_trues.append(cate.detach().cpu())

        cate_preds = torch.cat(cate_preds, dim=0).numpy()
        cate_trues = torch.cat(cate_trues, dim=0).numpy()

        train_ce_loss = ce_loss / len(train_loader)
        train_cate_acc = accuracy_score(cate_trues, cate_preds)
        train_cate_f1 = f1_score(cate_trues, cate_preds, average='macro')

        with torch.no_grad():
            backbone.eval()
            classifier.eval()

            ce_loss = 0.0
            cate_preds, cate_trues = [], []
            for step, (wv, cate, dist, dire) in enumerate(test_loader):
                wv, cate = wv.cuda(), cate.cuda()

                features = backbone(wv)
                cate_out = classifier(features)
                cate_loss = ce(cate_out, cate)
                cate_pred = torch.argmax(cate_out, dim=-1)

                cate_preds.append(cate_pred.cpu())
                cate_trues.append(cate.cpu())

                ce_loss += cate_loss.item()

            cate_preds = torch.cat(cate_preds, dim=0).numpy()
            cate_trues = torch.cat(cate_trues, dim=0).numpy()

            test_ce_loss = ce_loss / len(test_loader)
            test_cate_acc = accuracy_score(cate_trues, cate_preds)
            test_cate_f1 = f1_score(cate_trues, cate_preds, average='macro')

            print(epoch, f'train ce:{train_ce_loss:.4f}, train cate-acc:{train_cate_acc:.4f}, '
                         f'train cate-F1:{train_cate_f1:.4f} '
                         f'test ce:{test_ce_loss:.4f}, test cate-acc:{test_cate_acc:.4f}, '
                         f'test cate-F1:{test_cate_f1:.4f},')





