import torch
import torch.nn.functional as F
from classes.ShahedNet import ShahedNet
from classes.ShahedDataset import label2idx


def predict_full_audio_by_segments(model, file_path, device, segment_length=16000):
    model.eval()
    waveform = model.load_wav(file_path).to(device)  # [samples]
    total_length = waveform.size(0)

    results = []

    # Розбиваємо сигнал на сегменти по 1 секунді
    for start in range(0, total_length, segment_length):
        end = start + segment_length
        segment = waveform[start:end]

        # Якщо сегмент коротший 1 секунди — доповнюємо нулями
        if segment.size(0) < segment_length:
            pad_size = segment_length - segment.size(0)
            segment = torch.cat([segment, torch.zeros(pad_size, device=device)])

        segment = segment.unsqueeze(0)  # [1, samples]

        with torch.no_grad():
            logits = model(segment)  # [1, n_classes]
            probs = torch.sigmoid(logits).cpu().numpy()[0]  # вірогідності класів

        results.append(probs)

    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ShahedNet(n_classes=len(label2idx))
    model.load_state_dict(torch.load("saved_models/shahednet_best_light.pth", map_location=device))
    model.to(device)

    # test_file = "audio_for_test/from_dataset_shahed2.wav"
    test_file = "audio_for_test/from_dataset_audioWind3.wav"
    probs_list = predict_full_audio_by_segments(model, test_file, device)

    idx2label = {v: k for k, v in label2idx.items()}

    print(f"Результати передбачення для файлу {test_file}:")
    for i, probs in enumerate(probs_list):
        print(f" Сегмент {i + 1}:")
        for j, p in enumerate(probs):
            print(f"  Клас '{idx2label[j]}': впевненість {p:.4f}")

if __name__ == "__main__":
    main()
