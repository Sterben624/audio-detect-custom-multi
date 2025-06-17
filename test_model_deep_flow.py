import torch
import sounddevice as sd
import numpy as np
from torchaudio.transforms import Resample

from classes.ShahedNetDeepWide import ShahedNetDeepWide
from classes.ShahedDataset import label2idx


def predict_segment(model, segment, device):
    model.eval()
    tensor_segment = torch.tensor(segment, dtype=torch.float32).to(device)
    tensor_segment = tensor_segment.unsqueeze(0)  # [1, samples]

    with torch.no_grad():
        logits = model(tensor_segment)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    return probs


def record_and_predict(model, device, input_rate=44100, target_rate=16000, segment_duration=1.0, threshold=0.8):
    idx2label = {v: k for k, v in label2idx.items()}
    shahed_idx = label2idx.get("shahed", None)
    input_samples = int(input_rate * segment_duration)

    resampler = Resample(orig_freq=input_rate, new_freq=target_rate)

    print("Слухаю мікрофон... Натисніть Ctrl+C для зупинки.")

    try:
        while True:
            audio = sd.rec(input_samples, samplerate=input_rate, channels=1, dtype='float32')
            sd.wait()
            audio = audio.flatten()  # [samples]

            waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            resampled = resampler(waveform).squeeze(0).numpy()

            probs = predict_segment(model, resampled, device)

            dominant_idx = int(np.argmax(probs))
            dominant_label = idx2label[dominant_idx]
            dominant_conf = probs[dominant_idx]

            print("-" * 30)
            print(f"Домінуючий клас: '{dominant_label}' з впевненістю {dominant_conf:.4f}")
            print("Ймовірності по класах:")

            for idx, prob in enumerate(probs):
                label = idx2label[idx]
                print(f"  Клас '{label}': {prob:.4f}")

            if shahed_idx is not None and probs[shahed_idx] >= threshold:
                print("\n⚠️  Виявлено 'shahed' (впевненість >= 0.8)")
            else:
                print("\n'shahed' не виявлено")
    except KeyboardInterrupt:
        print("\nЗупинено користувачем.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ShahedNetDeepWide(n_classes=len(label2idx))
    model.load_state_dict(torch.load("saved_models/shahednet_best.pth", map_location=device))
    model.to(device)

    record_and_predict(model, device)


if __name__ == "__main__":
    main()
