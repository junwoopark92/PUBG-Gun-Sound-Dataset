import glob
import os
# os.environ['CUDA_VISIBLE_DEVICES']='4'
import shutil
import random

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

from dataset import ForenGunDataset, BGGunDataset, UrbanGunDataset
from models import CNNExtractor, RNNExtractor, CRNNExtractor, TransformerExtractor, CTransExtractor, SingleClassifer


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))


if __name__ == '__main__':
    print(torch.__version__)
    print(torchaudio.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='CNN', type=str, choices=['CNN', 'RNN', 'CRNN', 'Trans', 'CTrans'])
    parser.add_argument('--pretrained', default=None, type=str)
    parser.add_argument('--dataset', default='BGG', type=str, choices=['BGG', 'Foren', 'Urban'])
    parser.add_argument('--datadir', default='./data/gun_sound_v1', type=str)
    parser.add_argument('--input_sec', default=2, type=int)
    parser.add_argument('--sr', default=44100, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--n_train', default=None, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--hdim', default=64, type=int)

    parser.add_argument('--use_bgg', default=False, action='store_true')
    parser.add_argument('--save_midi', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    print(args)

    # load dataset
    if args.dataset == 'BGG':
        train_dataset = BGGunDataset(args.datadir, './data/exp1_train.csv', args.sr, args.input_sec, 'train')
        val_dataset = None
        test_dataset = BGGunDataset(args.datadir, './data/exp1_test.csv', args.sr, args.input_sec, 'test',
                                       dicts=train_dataset.dicts)
    elif args.dataset == 'Foren':
        train_dataset = ForenGunDataset(args.datadir, './data/GunshotAudioForensicsDataset/train.csv', args.sr, args.input_sec, 'train',
                                        n_train=args.n_train)
        val_dataset = ForenGunDataset(args.datadir, './data/GunshotAudioForensicsDataset/val.csv', args.sr, args.input_sec, 'val',
                                       dicts=train_dataset.dicts)
        test_dataset = ForenGunDataset(args.datadir, './data/GunshotAudioForensicsDataset/test.csv', args.sr, args.input_sec, 'test',
                                        dicts=train_dataset.dicts)
    elif args.dataset == 'Urban':
        folds = list(range(1, 11))
        random.shuffle(folds)
        train_folds = folds[:6]
        val_folds = [folds[6]]
        test_folds = folds[7:]
        print(train_folds, val_folds, test_folds)

        train_dataset = UrbanGunDataset(args.datadir, '/home/junwoopark/UrbanSound8K/metadata/UrbanSound8K_BGG.csv', args.sr, args.input_sec, 'train',
                                        train_folds, use_bgg=args.use_bgg)
        val_dataset = UrbanGunDataset(args.datadir, '/home/junwoopark/UrbanSound8K/metadata/UrbanSound8K.csv', args.sr, args.input_sec, 'val',
                                      val_folds, dicts=train_dataset.dicts)
        test_dataset = UrbanGunDataset(args.datadir, '/home/junwoopark/UrbanSound8K/metadata/UrbanSound8K.csv', args.sr, args.input_sec, 'test',
                                       test_folds, dicts=train_dataset.dicts)
    else:
        raise Exception(f'Not supported dataset{args.dataset}')

    label_dicts = train_dataset.dicts
    n_class = len(label_dicts['cate']) if args.dataset != 'Urban' else 2

    print(f'EXP1 for {args.dataset} Training:{len(train_dataset)} Test:{len(test_dataset)}, # of labels: {n_class}')
    print()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

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

    savepath = f'{args.dataset}-{args.backbone}-{args.hdim}-{args.input_sec}s-backbone.pt'

    if args.pretrained:
        load_model(backbone, args.pretrained)
        print(f'pretrained from {args.pretrained}')
        savepath = 'from-pretrained-'+savepath

    classifier = SingleClassifer(args.hdim, n_class).cuda()
    print(backbone)
    print(classifier)
    params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f'# of Backbone Parameters:{params}')

    ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(list(backbone.parameters()) + list(classifier.parameters()), lr=args.lr)

    best_val_ce = 1000
    best_results = []
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

            val_cate_f1, val_ce_loss, val_cate_acc = 0, 0, 0
            if val_dataset:
                ce_loss = 0.0
                cate_preds, cate_trues = [], []
                for step, (wv, cate, dist, dire) in enumerate(val_loader):
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

                val_ce_loss = ce_loss / len(val_loader)
                val_cate_acc = accuracy_score(cate_trues, cate_preds)
                val_cate_f1 = f1_score(cate_trues, cate_preds, average='macro')

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

            if args.dataset == 'BGG':
                val_ce_loss = test_ce_loss

            if best_val_ce >= val_ce_loss:
                best_val_ce = val_ce_loss
                best_results = [str(i) for i in [epoch, train_ce_loss, train_cate_acc, train_cate_f1,
                                val_ce_loss, val_cate_acc, val_cate_f1,
                                test_ce_loss, test_cate_acc, test_cate_f1]]
                best_results = '\t'.join(best_results)

            print(epoch, f'train ce:{train_ce_loss:.4f}, train cate-acc:{train_cate_acc:.4f}, '
                         f'train cate-F1:{train_cate_f1:.4f}'
                         f' val ce:{val_ce_loss:.4f}, val cate-acc:{val_cate_acc:.4f}, '
                         f'val cate-F1:{val_cate_f1:.4f}'  
                         f' test ce:{test_ce_loss:.4f}, test cate-acc:{test_cate_acc:.4f},'
                         f'test cate-F1:{test_cate_f1:.4f}')

    save_model(backbone, savepath)
    print(f"best results: {best_results}")
    print(f'backbone saved at {savepath}')




