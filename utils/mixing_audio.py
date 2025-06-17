import os
import random
import torchaudio
import torch
import csv

# Параметри
input_dirs = [
    "dataset/to_train/siren",
    "dataset/to_train/speech",
    "dataset/to_train/wind",
    "dataset/to_train/self",
    "dataset/to_train/shahed",
    "dataset/to_train/explosion"
]
output_dir = "dataset/mixed_train"
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "labels.csv")

sample_rate = 16000  # Ваш фіксований sr
duration_sec = 1  # Часова довжина аудіо (секунда)
samples_per_clip = sample_rate * duration_sec

# Отримаємо список файлів у кожній папці
files_by_class = {}
for d in input_dirs:
    label = os.path.basename(d)
    files_by_class[label] = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".wav")]

# Функція для завантаження та обрізання/доповнення аудіо до 1 секунди
def load_fixed_length_wav(path, sr=sample_rate, length=samples_per_clip):
    waveform, file_sr = torchaudio.load(path)
    if file_sr != sr:
        waveform = torchaudio.transforms.Resample(file_sr, sr)(waveform)
    waveform = waveform[0]  # вибираємо 1 канал
    if waveform.size(0) > length:
        waveform = waveform[:length]
    elif waveform.size(0) < length:
        padding = length - waveform.size(0)
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    return waveform

# Генерація міксованих файлів
num_mixed_files = 3000  # Кількість згенерованих міксованих аудіо

with open(csv_path, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["filename", "labels"])  # Заголовки

    for i in range(num_mixed_files):
        # Випадково вибираємо 2 або 3 класи без повторів
        selected_classes = random.sample(files_by_class.keys(), random.choice([2, 3]))

        # Завантажуємо випадковий файл для кожного класу
        waveforms = []
        for cls in selected_classes:
            file_path = random.choice(files_by_class[cls])
            wav = load_fixed_length_wav(file_path)
            waveforms.append(wav)

        # Міксуємо - сумуємо сигнали
        mixed_waveform = sum(waveforms)

        # Нормалізуємо сигнал (щоб не було кліпінгу)
        max_val = mixed_waveform.abs().max()
        if max_val > 1.0:
            mixed_waveform = mixed_waveform / max_val

        # Зберігаємо у файл
        filename = f"mixed_{i:04d}.wav"
        output_path = os.path.join(output_dir, filename)
        torchaudio.save(output_path, mixed_waveform.unsqueeze(0), sample_rate)

        # Записуємо в CSV
        labels_str = ",".join(selected_classes)
        csv_writer.writerow([filename, labels_str])

print("Генерація міксованих аудіо завершена.")
