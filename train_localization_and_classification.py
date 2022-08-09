import glob
import os
import warnings
warnings.filterwarnings('ignore')
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

from dataset import BGGunDataset, ForenGunDataset
from models import CNNExtractor, RNNExtractor, CRNNExtractor, TransformerExtractor, CTransExtractor, MultitaskClassifer, \
    OurClassifer, OurExtractor


def load_model(model, path):
    model.load_state_dict(torch.load(path))


if __name__ == '__main__':
    print(torch.__version__)
    print(torchaudio.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='CNN', type=str, choices=['CNN', 'RNN', 'CRNN', 'Trans', 'CTrans','Our'])
    parser.add_argument('--pretrained', default=None, type=str)
    parser.add_argument('--dataset', default='BGG', type=str, choices=['BGG', 'Foren', 'Urban'])
    parser.add_argument('--datadir', default='./data/gun_sound_v2', type=str)
    parser.add_argument('--input_sec', default=3, type=int)
    parser.add_argument('--sr', default=44100, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--hdim', default=64, type=int)
    parser.add_argument('--save', default=None, type=str)

    parser.add_argument('--save_midi', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    print(args)

    # load dataset
    if args.dataset == 'BGG':
        a = pd.read_csv('./data/v3_exp1_train.csv')
        b = pd.read_csv('./data/v3_exp2_train.csv')
        train_df = pd.concat([a, b], axis=0)
        #train_df = b
        a = pd.read_csv('./data/v3_exp1_test.csv')
        b = pd.read_csv('./data/v3_exp2_test.csv')
        #test_df = b
        test_df = pd.concat([a, b], axis=0)
        train_dataset = BGGunDataset(args.datadir, train_df, args.sr, args.input_sec, 'train')
        val_dataset = BGGunDataset(args.datadir, train_dataset.get_val_df(), args.sr, args.input_sec, 'val',
                                   dicts=train_dataset.dicts)
        test_dataset = BGGunDataset(args.datadir, test_df, args.sr, args.input_sec, 'test',
                                    dicts=train_dataset.dicts)


    elif args.dataset == 'Foren':
        train_dataset = ForenGunDataset(args.datadir, './data/GunshotAudioForensicsDataset/train_v2.csv', args.sr, args.input_sec, 'train',
                                        )
        val_dataset = ForenGunDataset(args.datadir, './data/GunshotAudioForensicsDataset/val_v2.csv', args.sr, args.input_sec, 'val',
                                      dicts=train_dataset.dicts)
        test_dataset = ForenGunDataset(args.datadir, './data/GunshotAudioForensicsDataset/test_v2.csv', args.sr, args.input_sec, 'test',
                                       dicts=train_dataset.dicts)

    label_dicts = train_dataset.dicts
    print(label_dicts)
    print(f'EXP2 Gunshot Localization and Classification Training:{len(train_dataset)}, Val:{len(val_dataset)}, Test:{len(test_dataset)}')
    print()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
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
    elif args.backbone == 'Our':
        backbone = OurExtractor(args.hdim, sample_rate=args.sr, n_fft=512, n_mels=96).cuda()
    else:
        raise Exception(f'Not Supported Backbone:{args.backbone}')

    if args.pretrained:
        load_model(backbone, args.pretrained)
        print(f'pretrained from {args.pretrained}')
        # savepath = 'from-pretrained-'+ savepath

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

    ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(list(backbone.parameters()) + list(classifier.parameters()), lr=args.lr)

    best_val_ce = 1000
    for epoch in range(args.epochs):
        backbone.train()
        classifier.train()

        ce_loss = 0.0
        cate_preds, cate_trues, dist_preds, dist_trues, dire_preds, dire_trues = [], [], [], [], [], []
        for step, (wv, cate, dist, dire) in enumerate(train_loader):
            wv, cate, dist, dire = wv.cuda(), cate.cuda(), dist.cuda(), dire.cuda()

            features = backbone(wv)
            cate_out, dist_out, dire_out = classifier(features)
            cate_loss = ce(cate_out, cate)
            dist_loss = ce(dist_out, dist)
            dire_loss = ce(dire_out, dire)

            cate_pred = torch.argmax(cate_out, dim=-1)
            dist_pred = torch.argmax(dist_out, dim=-1)
            dire_pred = torch.argmax(dire_out, dim=-1)

            loss = cate_loss + dist_loss + dire_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            ce_loss += loss.item()

            cate_preds.append(cate_pred.detach().cpu())
            cate_trues.append(cate.detach().cpu())
            dist_preds.append(dist_pred.detach().cpu())
            dist_trues.append(dist.detach().cpu())
            dire_preds.append(dire_pred.detach().cpu())
            dire_trues.append(dire.detach().cpu())

        cate_preds = torch.cat(cate_preds, dim=0).numpy()
        cate_trues = torch.cat(cate_trues, dim=0).numpy()
        dist_preds = torch.cat(dist_preds, dim=0).numpy()
        dist_trues = torch.cat(dist_trues, dim=0).numpy()
        dire_preds = torch.cat(dire_preds, dim=0).numpy()
        dire_trues = torch.cat(dire_trues, dim=0).numpy()

        train_ce_loss = ce_loss / len(train_loader)
        train_cate_acc = accuracy_score(cate_trues, cate_preds)
        train_cate_f1 = f1_score(cate_trues, cate_preds, average='macro')
        train_dist_acc = accuracy_score(dist_trues, dist_preds)
        train_dist_f1 = f1_score(dist_trues, dist_preds, average='macro')
        train_dire_acc = accuracy_score(dire_trues, dire_preds)
        train_dire_f1 = f1_score(dire_trues, dire_preds, average='macro')

        with torch.no_grad():
            backbone.eval()
            classifier.eval()

            ce_loss = 0.0
            cate_preds, cate_trues, dist_preds, dist_trues, dire_preds, dire_trues = [], [], [], [], [], []
            for step, (wv, cate, dist, dire) in enumerate(val_loader):
                wv, cate, dist, dire = wv.cuda(), cate.cuda(), dist.cuda(), dire.cuda()

                features = backbone(wv)
                cate_out, dist_out, dire_out = classifier(features)
                cate_loss = ce(cate_out, cate)
                dist_loss = ce(dist_out, dist)
                dire_loss = ce(dire_out, dire)

                loss = cate_loss + dist_loss + dire_loss

                cate_pred = torch.argmax(cate_out, dim=-1)
                dist_pred = torch.argmax(dist_out, dim=-1)
                dire_pred = torch.argmax(dire_out, dim=-1)

                ce_loss += loss.item()

                cate_preds.append(cate_pred.cpu())
                cate_trues.append(cate.cpu())
                dist_preds.append(dist_pred.cpu())
                dist_trues.append(dist.cpu())
                dire_preds.append(dire_pred.cpu())
                dire_trues.append(dire.cpu())

            cate_preds = torch.cat(cate_preds, dim=0).numpy()
            cate_trues = torch.cat(cate_trues, dim=0).numpy()
            dist_preds = torch.cat(dist_preds, dim=0).numpy()
            dist_trues = torch.cat(dist_trues, dim=0).numpy()
            dire_preds = torch.cat(dire_preds, dim=0).numpy()
            dire_trues = torch.cat(dire_trues, dim=0).numpy()

            val_ce_loss = ce_loss / len(test_loader)
            val_cate_acc = accuracy_score(cate_trues, cate_preds)
            val_cate_f1 = f1_score(cate_trues, cate_preds, average='macro')
            val_dist_acc = accuracy_score(dist_trues, dist_preds)
            val_dist_f1 = f1_score(dist_trues, dist_preds, average='macro')
            val_dire_acc = accuracy_score(dire_trues, dire_preds)
            val_dire_f1 = f1_score(dire_trues, dire_preds, average='macro')

            ce_loss = 0.0
            cate_preds, cate_trues, dist_preds, dist_trues, dire_preds, dire_trues = [], [], [], [], [], []
            for step, (wv, cate, dist, dire) in enumerate(test_loader):
                wv, cate, dist, dire = wv.cuda(), cate.cuda(), dist.cuda(), dire.cuda()

                features = backbone(wv)
                cate_out, dist_out, dire_out = classifier(features)
                cate_loss = ce(cate_out, cate)
                dist_loss = ce(dist_out, dist)
                dire_loss = ce(dire_out, dire)

                loss = cate_loss + dist_loss + dire_loss

                cate_pred = torch.argmax(cate_out, dim=-1)
                dist_pred = torch.argmax(dist_out, dim=-1)
                dire_pred = torch.argmax(dire_out, dim=-1)

                ce_loss += loss.item()

                cate_preds.append(cate_pred.cpu())
                cate_trues.append(cate.cpu())
                dist_preds.append(dist_pred.cpu())
                dist_trues.append(dist.cpu())
                dire_preds.append(dire_pred.cpu())
                dire_trues.append(dire.cpu())

            cate_preds = torch.cat(cate_preds, dim=0).numpy()
            cate_trues = torch.cat(cate_trues, dim=0).numpy()
            dist_preds = torch.cat(dist_preds, dim=0).numpy()
            dist_trues = torch.cat(dist_trues, dim=0).numpy()
            dire_preds = torch.cat(dire_preds, dim=0).numpy()
            dire_trues = torch.cat(dire_trues, dim=0).numpy()

            test_ce_loss = ce_loss / len(test_loader)
            test_cate_acc = accuracy_score(cate_trues, cate_preds)
            test_cate_f1 = f1_score(cate_trues, cate_preds, average='macro')
            test_dist_acc = accuracy_score(dist_trues, dist_preds)
            test_dist_f1 = f1_score(dist_trues, dist_preds, average='macro')
            test_dire_acc = accuracy_score(dire_trues, dire_preds)
            test_dire_f1 = f1_score(dire_trues, dire_preds, average='macro')

            print(epoch, f'train ce:{train_ce_loss:.4f}, '
                         f'train cate-acc:{train_cate_acc:.4f}, train cate-F1:{train_cate_f1:.4f}, '
                         f'train dist-acc:{train_dist_acc:.4f}, train dist-F1:{train_dist_f1:.4f}, '
                         f'train dire-acc:{train_dire_acc:.4f}, train dire-F1:{train_dire_f1:.4f} '
                         f'val ce:{val_ce_loss:.4f}, '
                         f'val cate-acc:{val_cate_acc:.4f}, val cate-F1:{val_cate_f1:.4f}, '
                         f'val dist-acc:{val_dist_acc:.4f}, val dist-F1:{val_dist_f1:.4f}, '
                         f'val dire-acc:{val_dire_acc:.4f}, val dire-F1:{val_dire_f1:.4f} '
                         f'test ce:{test_ce_loss:.4f}, '
                         f'test cate-acc:{test_cate_acc:.4f}, test cate-F1:{test_cate_f1:.4f}, ' 
                         f'test dist-acc:{test_dist_acc:.4f}, test dist-F1:{test_dist_f1:.4f}, '
                         f'test dire-acc:{train_dire_acc:.4f}, test dire-F1:{test_dire_f1:.4f}')

            if best_val_ce >= val_ce_loss:
                best_val_ce = val_ce_loss
                best_results = [str(i) for i in [epoch, train_ce_loss, train_cate_acc, train_cate_f1,
                                                 train_dist_acc, train_dist_f1, train_dire_acc, train_dire_f1,
                                                 val_ce_loss, val_cate_acc, val_cate_f1,
                                                 val_dist_acc, val_dist_f1, val_dire_acc, val_dire_f1,
                                                 test_ce_loss, test_cate_acc, test_cate_f1,
                                                 test_dist_acc, test_dist_f1, train_dire_acc, test_dire_f1]]

                best_results = '\t'.join(best_results)
                if args.dataset == 'BGG':
                    torch.save(backbone.state_dict(), f'demo_{args.backbone}_backbone.pt')
                    torch.save(classifier.state_dict(), f'demo_{args.backbone}_classifier.pt')
    print(best_results)




