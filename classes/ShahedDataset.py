from torchaudio import load
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import pandas as pd
import os

class ShahedDataset(Dataset):
    def __init__(self, csv_path, audio_dir, label2idx, target_sr=16000):
        self.data = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.label2idx = label2idx
        self.target_sr = target_sr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = row['filename']
        labels_str = row['labels']  # Наприклад "speech,wind"

        waveform, sr = torchaudio.load(os.path.join(self.audio_dir, filename))
        if sr != self.target_sr:
            waveform = T.Resample(sr, self.target_sr)(waveform)

        # Формуємо multi-hot вектор
        labels = torch.zeros(len(self.label2idx), dtype=torch.float32)
        for lbl in labels_str.split(","):
            lbl = lbl.strip()
            if lbl in self.label2idx:
                labels[self.label2idx[lbl]] = 1.0

        return waveform[0], labels  # [samples], [multi-hot]

# Мапа класів. Залишу поки тут
label2idx = {
    "speech": 0,
    "wind": 1,
    "self": 2,
    "shahed": 3,
    "siren": 4,
    "explosion": 5
}
