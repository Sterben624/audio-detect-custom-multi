import torch
import torchaudio
import os

def load_and_trim(path, target_len=16000, sr=16000):
    waveform, sample_rate = torchaudio.load(path)
    if sample_rate != sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sr)
        waveform = resampler(waveform)

    waveform = waveform[0]  # моно

    if waveform.size(0) < target_len:
        pad_size = target_len - waveform.size(0)
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))
    elif waveform.size(0) > target_len:
        start = (waveform.size(0) - target_len) // 2
        waveform = waveform[start:start + target_len]

    return waveform

def predict(model, wav_path, device, label2idx):
    model.eval()
    model = model.to(device)
    waveform = load_and_trim(wav_path).to(device)
    with torch.no_grad():
        output = model(waveform.unsqueeze(0))
        predicted_idx = output.argmax(dim=1).item()

    idx2label = {v: k for k, v in label2idx.items()}
    return idx2label[predicted_idx]

def batch_predict(model, folder_path, device, label2idx):
    results = {}
    for fname in os.listdir(folder_path):
        if fname.endswith(".wav"):
            path = os.path.join(folder_path, fname)
            label = predict(model, path, device, label2idx)
            results[fname] = label
    return results
