import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T


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
        # Predict tag using the aggregated features.

    def forward(self, x):
        x = self.spec(x)
        x = self.to_db(x)
        x = self.spec_bn(x)
        B, C, M, T = x.shape
        x = x.reshape(B, C*M, T) # for 1D conv
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.final_pool(x)
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

    def forward(self):
        pass


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

    def forward(self):
        pass


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

    def forward(self):
        pass


class SingleClassifer(nn.Module):
    def __init__(self, hidden_dim, n_class):
        super(SingleClassifer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_class))

    def forward(self, feature):
        logit = self.fc(feature.squeeze(-1))
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