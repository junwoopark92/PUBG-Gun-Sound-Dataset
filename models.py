import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

from transformer import PositionalEncoding, TransformerEncoderBlock


class MeanCNNExtractor(nn.Module):
    def __init__(self, hidden_dim,
                 sample_rate=16000,
                 n_fft=512,
                 f_min=0.0,
                 f_max=8000.0,
                 n_mels=96):
        """
        Args:
          sample_rate (int): path to load dataset from
          n_fft (int): number of samples for fft
          f_min (float): min freq
          f_max (float): max freq
          n_mels (float): number of mel bin
          n_class (int): number of class
        """
        super(MeanCNNExtractor, self).__init__()
        # Spectrogram

        self.conv0 = nn.Sequential(
            nn.Conv2d(1, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels=hidden_dim*2, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # 1, 16, 8
        # print(x.shape)
        x = self.conv0(x)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.dropout(x)
        return x



class CNNExtractor(nn.Module):
    def __init__(self, hidden_dim,
                 sample_rate=16000,
                 n_fft=512,
                 f_min=0.0,
                 f_max=8000.0,
                 n_mels=96):
        """
        Args:
          sample_rate (int): path to load dataset from
          n_fft (int): number of samples for fft
          f_min (float): min freq
          f_max (float): max freq
          n_mels (float): number of mel bin
          n_class (int): number of class
        """
        super(CNNExtractor, self).__init__()
        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(2)

        self.conv0 = nn.Sequential(
            nn.Conv1d(n_mels*2, out_channels=hidden_dim, kernel_size=3, stride=1, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        # Aggregate features over temporal dimension.
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.2)
        # Predict tag using the aggregated features.

    def forward(self, x):
        # B, 2, T
        x = self.spec(x)
        # B, T, M  (B, 192, 96) > (B, 96)
        x = self.to_db(x)
        x = self.spec_bn(x)
        B, C, M, T = x.shape
        x = x.reshape(B, C*M, T) # for 1D conv
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.final_pool(x)
        x = self.dropout(x)
        return x


class CRNNExtractor(nn.Module):
    def __init__(self, hidden_dim,
                 sample_rate=16000,
                 n_fft=512,
                 f_min=0.0,
                 f_max=8000.0,
                 n_mels=96):
        """
        Args:
          sample_rate (int): path to load dataset from
          n_fft (int): number of samples for fft
          f_min (float): min freq
          f_max (float): max freq
          n_mels (float): number of mel bin
          n_class (int): number of class
        """
        super(CRNNExtractor, self).__init__()
        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(2)

        self.conv0 = nn.Sequential(
            nn.Conv1d(n_mels*2, out_channels=hidden_dim, kernel_size=3, stride=1, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        # Predict tag using the aggregated features.

        self.n_layers = 2
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim//self.n_layers, num_layers=self.n_layers,
                            bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.spec(x)
        x = self.to_db(x)
        x = self.spec_bn(x)
        B, C, M, T = x.shape
        x = x.reshape(B, C*M, T) # for 1D conv
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1).contiguous()
        x, (hn, cn) = self.lstm(x)
        x = x.mean(dim=1)
        return x


class RNNExtractor(nn.Module):
    def __init__(self, hidden_dim,
                 sample_rate=16000,
                 n_fft=512,
                 f_min=0.0,
                 f_max=8000.0,
                 n_mels=96):
        """
        Args:
          sample_rate (int): path to load dataset from
          n_fft (int): number of samples for fft
          f_min (float): min freq
          f_max (float): max freq
          n_mels (float): number of mel bin
          n_class (int): number of class
        """
        super(RNNExtractor, self).__init__()
        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(2)

        self.n_layers = 2
        self.lstm = nn.LSTM(input_size=192, hidden_size=hidden_dim//self.n_layers, num_layers=self.n_layers,
                            bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim*2, hidden_dim)

    def forward(self, x):
        x = self.spec(x)
        x = self.to_db(x)
        x = self.spec_bn(x)
        B, C, M, T = x.shape
        x = x.reshape(B, C*M, T)  # for 1D
        x = x.permute(0, 2, 1).contiguous()
        x, (hn, cn) = self.lstm(x)
        # print(x.shape, hn.shape)
        x = x.mean(dim=1)
        # hn = hn.permute(1, 0, 2).reshape(B, -1)
        # x = self.fc(hn)
        return x


class TransformerExtractor(nn.Module):
    def __init__(self, hidden_dim,
                 sample_rate=16000,
                 n_fft=512,
                 f_min=0.0,
                 f_max=8000.0,
                 n_mels=96):
        """
        Args:
          sample_rate (int): path to load dataset from
          n_fft (int): number of samples for fft
          f_min (float): min freq
          f_max (float): max freq
          n_mels (float): number of mel bin
          n_class (int): number of class
        """
        super(TransformerExtractor, self).__init__()
        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(2)

        self.position_encoder = PositionalEncoding(192, 520)
        self.dropout = nn.Dropout(0.5)
        self.num_layers = 1

        encoders = [TransformerEncoderBlock(hidden_dim=192, dropout=0.5, n_head=4, feed_forward_dim=hidden_dim) \
                    for _ in range(self.num_layers)]
        self.encoders = nn.ModuleList(encoders)

        self.fc = nn.Linear(192, hidden_dim)

    def forward(self, x):
        """
            Paramters:
            x -- Input tensor
                    in shape (sequence_length, batch_size, hidden_dim)
            padding_mask -- Padding mask tensor in torch.bool type
                    in shape (sequence_length, batch_size)
                    True for <PAD>, False for non-<PAD>

            Returns:
            output -- output tensor
            in shape (sequence_length, batch_size, hidden_dim)
        """

        x = self.spec(x)
        x = self.to_db(x)
        x = self.spec_bn(x)
        B, C, M, T = x.shape
        x = x.reshape(B, C*M, T)  # for 1D
        x = x.permute(2, 0, 1).contiguous()
        padding_mask = torch.zeros(T, B).bool().cuda()
        out = self.position_encoder(x)
        out = self.dropout(out)
        for encoder in self.encoders:
            out = encoder(out, padding_mask=padding_mask)
        # L, B, C
        x = self.fc(out.mean(dim=0))
        return x

class CTransExtractor(nn.Module):
    def __init__(self, hidden_dim,
                 sample_rate=16000,
                 n_fft=512,
                 f_min=0.0,
                 f_max=8000.0,
                 n_mels=96):
        """
        Args:
          sample_rate (int): path to load dataset from
          n_fft (int): number of samples for fft
          f_min (float): min freq
          f_max (float): max freq
          n_mels (float): number of mel bin
          n_class (int): number of class
        """
        super(CTransExtractor, self).__init__()
        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(2)

        self.conv0 = nn.Sequential(
            nn.Conv1d(n_mels*2, out_channels=hidden_dim, kernel_size=3, stride=1, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        # Predict tag using the aggregated features.

        self.position_encoder = PositionalEncoding(hidden_dim, 25)
        self.dropout = nn.Dropout(0.5)
        self.num_layers = 1

        encoders = [TransformerEncoderBlock(hidden_dim=hidden_dim, dropout=0.5, n_head=4, feed_forward_dim=hidden_dim) \
                    for _ in range(self.num_layers)]
        self.encoders = nn.ModuleList(encoders)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.spec(x)
        x = self.to_db(x)
        x = self.spec_bn(x)
        B, C, M, T = x.shape
        x = x.reshape(B, C*M, T) # for 1D conv
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = x.permute(2, 0, 1).contiguous()
        sl, _, _ = x.shape
        padding_mask = torch.zeros(sl, B).bool().cuda()
        out = self.position_encoder(x)
        out = self.dropout(out)
        for encoder in self.encoders:
            out = encoder(out, padding_mask=padding_mask)
        # L, B, C
        return out.mean(dim=0)


class SingleClassifer(nn.Module):
    def __init__(self, hidden_dim, n_class):
        super(SingleClassifer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, n_class))

    def forward(self, feature):
        logit = self.fc(feature.flatten(start_dim=1))
        return logit


class MultitaskClassifer(nn.Module):
    def __init__(self, hidden_dim, n_cate, n_dist, n_direction):
        super(MultitaskClassifer, self).__init__()
        self.cate_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_cate))
        self.dist_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_dist))
        self.dire_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_direction))

    def forward(self, feature):
        cate_logit = self.cate_fc(feature.squeeze(-1))
        dist_logit = self.dist_fc(feature.squeeze(-1))
        dire_logit= self.dire_fc(feature.squeeze(-1))
        return cate_logit, dist_logit, dire_logit


class OurExtractor(nn.Module):
    def __init__(self, hidden_dim,
                 sample_rate=16000,
                 n_fft=512,
                 f_min=0.0,
                 f_max=8000.0,
                 n_mels=96):
        """
        Args:
          sample_rate (int): path to load dataset from
          n_fft (int): number of samples for fft
          f_min (float): min freq
          f_max (float): max freq
          n_mels (float): number of mel bin
          n_class (int): number of class
        """
        super(OurExtractor, self).__init__()
        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(2)

        self.conv0 = nn.Sequential(
            nn.Conv1d(n_mels*2, out_channels=hidden_dim, kernel_size=3, stride=1, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )


    def forward(self, x):
        x = self.spec(x)
        x = self.to_db(x)
        x = self.spec_bn(x)
        B, C, M, T = x.shape
        x = x.reshape(B, C*M, T) # for 1D conv
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x



class OurClassifer(nn.Module):
    def __init__(self, hidden_dim, n_cate, n_dist, n_direction):
        super(OurClassifer, self).__init__()

        # gun types: Transformer
        self.position_encoder = PositionalEncoding(hidden_dim, 25)
        self.dropout = nn.Dropout(0.5)
        self.num_layers = 1

        encoders = [TransformerEncoderBlock(hidden_dim=hidden_dim, dropout=0.5, n_head=4, feed_forward_dim=hidden_dim) \
                    for _ in range(self.num_layers)]
        self.encoders = nn.ModuleList(encoders)
        self.cate_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_cate))

        # distance: CRNN
        self.n_layers = 2
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim//self.n_layers, num_layers=self.n_layers,
                            bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim*2, hidden_dim)
        self.dist_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_dist))

        # CNN or Transformer
        self.dire_final_pool = nn.AdaptiveAvgPool1d(1)
        self.dire_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_direction))

    def forward(self, feature):

        B, _, _ = feature.shape
        x = feature.clone().permute(2, 0, 1).contiguous()
        padding_mask = torch.zeros(21, B).bool().cuda()
        out = self.position_encoder(x)
        out = self.dropout(out)
        for encoder in self.encoders:
            out = encoder(out, padding_mask=padding_mask)
        # L, B, C
        cate_logit = self.cate_fc(out.mean(dim=0))

        x = feature.clone().permute(0, 2, 1).contiguous()
        x, (hn, cn) = self.lstm(x)
        hn = hn.permute(1, 0, 2).reshape(B, -1)
        out = self.fc(hn)
        dist_logit = self.dist_fc(out)

        out = self.dire_final_pool(feature)
        dire_logit= self.dire_fc(out.squeeze(-1))
        return cate_logit, dist_logit, dire_logit

