import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ShahedNetDeepWide(nn.Module):
    def __init__(self, n_mels=64, n_classes=3, dropout_p=0.4):
        super().__init__()
        self.n_mels = n_mels

        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels
        )
        self.db_transform = T.AmplitudeToDB()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout2d(p=dropout_p)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.drop2 = nn.Dropout2d(p=dropout_p)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop3 = nn.Dropout2d(p=dropout_p)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.drop4 = nn.Dropout2d(p=dropout_p)

        self.se = SEBlock(256)

        self.pool = nn.MaxPool2d(2)

        self.rnn = nn.GRU(
            input_size=256,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.fc_drop = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(64 * 2, n_classes)

    def forward(self, waveform: torch.Tensor):
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.ndim == 2:
            mel = self.mel_transform(waveform)
            mel_db = self.db_transform(mel)
            x = mel_db.unsqueeze(1)  # [batch, 1, mel, time]
        else:
            raise ValueError("Очікується тензор розміром [samples] або [batch, samples]")

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.drop1(x)

        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.drop2(x)

        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.drop3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.drop4(x)

        x = self.se(x)  # SE attention

        # x: [B, C, H, W] → [B, T, C], де T = H*W
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        # GRU
        _, h_n = self.rnn(x)  # h_n: [2, B, 64]
        h_cat = torch.cat((h_n[0], h_n[1]), dim=1)  # [B, 128]

        x = self.fc_drop(h_cat)
        return self.fc2(x)

    @staticmethod
    def load_wav(path, sr=16000):
        waveform, sample_rate = torchaudio.load(path)
        if sample_rate != sr:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=sr)
            waveform = resampler(waveform)
        return waveform[0]
