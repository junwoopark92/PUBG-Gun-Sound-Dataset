import glob
import os
# os.environ['CUDA_VISIBLE_DEVICES']='4'
import shutil
import random
import warnings
warnings.filterwarnings('ignore')
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

from dataset import ForenGunDataset, BGGunDataset, UrbanGunDataset, UrbanGun1DDataset
from models import MeanCNNExtractor, CNNExtractor, RNNExtractor, CRNNExtractor, TransformerExtractor, CTransExtractor, SingleClassifer


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))


def inference(dataset, savename=None):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    save_features, cate_preds, cate_trues = [], [], []
    for step, (wv, cate, dist, dire) in enumerate(loader):
        wv, cate = wv.cuda(), cate.cuda()

        features = backbone(wv)
        cate_out = classifier(features)

        cate_pred = torch.argmax(cate_out, dim=-1)

        save_features.append(features.cpu())
        cate_preds.append(cate_pred.cpu())
        cate_trues.append(cate.cpu())

    save_features = torch.cat(save_features, dim=0).numpy()
    cate_preds = torch.cat(cate_preds, dim=0).numpy()
    cate_trues = torch.cat(cate_trues, dim=0).numpy()

    np.save(f'{savename}-features.npy', save_features)
    np.save(f'{savename}-preds.npy', cate_preds)
    np.save(f'{savename}-trues.npy', cate_trues)


if __name__ == '__main__':
    print(torch.__version__)
    print(torchaudio.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='CNN', type=str, choices=['CNN', 'RNN', 'CRNN', 'Trans', 'CTrans'])
    parser.add_argument('--pretrained', default=None, type=str)
    parser.add_argument('--dataset', default='BGG', type=str, choices=['BGG', 'Foren', 'Urban'])
    parser.add_argument('--datadir', default='./data/gun_sound_v2', type=str)
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
        df = pd.read_csv('./data/v3_exp1_train.csv')
        train_dataset = BGGunDataset(args.datadir, df, args.sr, args.input_sec, 'train')
        val_dataset = BGGunDataset(args.datadir, train_dataset.get_val_df(), args.sr, args.input_sec, 'val',
                                   dicts=train_dataset.dicts)
        test_dataset = BGGunDataset(args.datadir, pd.read_csv('./data/v3_exp1_test.csv'), args.sr, args.input_sec, 'test',
                                       dicts=train_dataset.dicts)
    elif args.dataset == 'Foren':
        train_dataset = ForenGunDataset(args.datadir, './data/GunshotAudioForensicsDataset/train.csv', args.sr, args.input_sec, 'train',
                                        n_train=args.n_train)
        val_dataset = ForenGunDataset(args.datadir, './data/GunshotAudioForensicsDataset/val.csv', args.sr, args.input_sec, 'val',
                                       dicts=train_dataset.dicts)
        test_dataset = ForenGunDataset(args.datadir, './data/GunshotAudioForensicsDataset/test.csv', args.sr, args.input_sec, 'test',
                                        dicts=train_dataset.dicts)
    elif args.dataset == 'Urban':
        df = pd.read_csv('/home/junwoopark/UrbanSound8K/metadata/UrbanSound8K.csv')
        df['aug'] = False
        nogun_df = df[df['classID'] != 6]
        gun_df = df[df['classID'] == 6]

        nogun_df = nogun_df.sample(len(nogun_df))
        n_train = int(len(nogun_df)*0.7)
        n_val = int(len(nogun_df)*0.1)

        n_gun_train = int(len(gun_df)*0.2)
        n_gun_val = int(len(gun_df)*0.4)

        nogun_train_df = nogun_df.iloc[:n_train]
        nogun_val_df = nogun_df.iloc[n_train: n_train+n_val]
        nogun_test_df = nogun_df.iloc[n_train+n_val:]

        gun_train_df = gun_df.iloc[:n_gun_train]
        gun_val_df = gun_df.iloc[n_gun_train: n_gun_train+n_gun_val]
        gun_test_df = gun_df.iloc[n_gun_train+n_gun_val:]

        train_df = pd.concat([nogun_train_df, gun_train_df], axis=0)
        val_df = pd.concat([nogun_val_df, gun_val_df], axis=0)
        test_df = pd.concat([nogun_test_df, gun_test_df], axis=0)

        bgg_df = pd.read_csv('/home/junwoopark/UrbanSound8K/metadata/UrbanSound8K_BGG.csv')
        bgg_df['aug'] = False

        print(f'use bgg: {len(bgg_df)}')
        n_bgg = len(bgg_df)
        aug_df = gun_train_df.sample(n_bgg, replace=True).copy()
        aug_df['aug'] = True
        # train_df = pd.concat([train_df, bgg_df, aug_df], axis=0)

        train_dataset = UrbanGunDataset(args.datadir, train_df, args.sr, args.input_sec, 'train',
                                        use_bgg=args.use_bgg)
        val_dataset = UrbanGunDataset(args.datadir, val_df, args.sr, args.input_sec, 'val',
                                      dicts=train_dataset.dicts)
        test_dataset = UrbanGunDataset(args.datadir, test_df, args.sr, args.input_sec, 'test',
                                       dicts=train_dataset.dicts)
        bgg_dataset = UrbanGunDataset(args.datadir, bgg_df, args.sr, args.input_sec, 'test',
                                      dicts=train_dataset.dicts)
        aug_dataset = UrbanGunDataset(args.datadir, aug_df, args.sr, args.input_sec, 'test',
                                      dicts=train_dataset.dicts)

    else:
        raise Exception(f'Not supported dataset{args.dataset}')

    label_dicts = train_dataset.dicts
    n_class = len(label_dicts['cate']) if args.dataset != 'Urban' else 10 #2

    print(f'EXP1 for {args.dataset} Training:{len(train_dataset)} Val:{len(val_dataset)} Test:{len(test_dataset)}, # of labels: {n_class}')
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
    cls_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f'# of Backbone Parameters:{params}, # of Classifier Parameters: {cls_params}')

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

            val_cate_f1, val_ce_loss, val_cate_acc, val_gun_acc = 0, 0, 0, 0
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
                gun_idx = (cate_trues == 6)
                val_gun_acc = accuracy_score(cate_trues[gun_idx], cate_preds[gun_idx])
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
            gun_idx = (cate_trues == 6)
            test_gun_acc = accuracy_score(cate_trues[gun_idx], cate_preds[gun_idx])
            test_cate_f1 = f1_score(cate_trues, cate_preds, average='macro')

            if best_val_ce >= val_ce_loss:
                best_val_ce = val_ce_loss
                best_results = [str(i) for i in [epoch, train_ce_loss, train_cate_acc, train_cate_f1,
                                val_ce_loss, val_cate_acc, val_cate_f1, val_gun_acc,
                                test_ce_loss, test_cate_acc, test_cate_f1, test_gun_acc]]
                best_results = '\t'.join(best_results)
                print(f'backbone saved at {savepath}')
                inference(test_dataset, 'test')
                inference(bgg_dataset, 'bgg')
                inference(aug_dataset, 'aug')

                # save_model(backbone, savepath)

            print(epoch, f'train ce:{train_ce_loss:.4f}, train cate-acc:{train_cate_acc:.4f}, '
                         f'train cate-F1:{train_cate_f1:.4f}'
                         f' val ce:{val_ce_loss:.4f}, val cate-acc:{val_cate_acc:.4f}, '
                         f'val cate-F1:{val_cate_f1:.4f}, val gun-acc:{val_gun_acc:.4f}'  
                         f' test ce:{test_ce_loss:.4f}, test cate-acc:{test_cate_acc:.4f},'
                         f'test cate-F1:{test_cate_f1:.4f}, test gun-acc:{test_gun_acc:.4f}')

    print(f"best results: {best_results}")




