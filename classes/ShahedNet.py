import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T


class ShahedNet(nn.Module):
    def __init__(self, n_mels=64, n_classes=3, dropout_p=0.3):
        super().__init__()
        self.n_mels = n_mels
        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels
        )
        self.db_transform = T.AmplitudeToDB()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=dropout_p)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, n_classes)

    def forward(self, waveform: torch.Tensor):
        """
        :param waveform: [batch, samples] або [samples]
        :return: logits [batch, classes]
        """
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # -> [1, samples]

        if waveform.ndim == 2:
            mel = self.mel_transform(waveform)  # -> [batch, mel, time]
            mel_db = self.db_transform(mel)
            x = mel_db.unsqueeze(1)  # [batch, 1, mel, time]
        else:
            raise ValueError("Очікується тензор розміром [samples] або [batch, samples]")

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

    @staticmethod
    def load_wav(path, sr=16000):
        waveform, sample_rate = torchaudio.load(path)
        if sample_rate != sr:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=sr)
            waveform = resampler(waveform)
        return waveform[0]  # прибираємо канал
