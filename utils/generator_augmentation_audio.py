import librosa
import soundfile as sf
import numpy as np
import os
from scipy.signal import butter, lfilter

def butter_filter(data, cutoff, fs, btype='low', order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return lfilter(b, a, data)

def augment_file(file_path, output_dir, num_augmented=4):
    y, sr = librosa.load(file_path, sr=None, mono=True)
    basename = os.path.splitext(os.path.basename(file_path))[0]

    for i in range(num_augmented):
        aug_y = y.copy()

        # Невеликий pitch shift
        n_steps = np.random.uniform(-1.5, 1.5)
        aug_y = librosa.effects.pitch_shift(aug_y, sr=sr, n_steps=n_steps)

        # Додавання шуму
        noise_amp = 0.002 * np.random.uniform(0.5, 1.0) * np.amax(aug_y)
        noise = noise_amp * np.random.normal(size=aug_y.shape)
        aug_y = aug_y + noise

        # Невелика фільтрація (низькочастотний фільтр)
        aug_y = butter_filter(aug_y, cutoff=3000, fs=sr, btype='low')

        # Нормалізація
        aug_y = aug_y / np.max(np.abs(aug_y))

        output_path = os.path.join(output_dir, f"{basename}_aug_{i}.wav")
        sf.write(output_path, aug_y, sr)

def augment_audio_folder(input_dir, output_dir, num_augmented=4):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.wav'):
            input_path = os.path.join(input_dir, filename)
            augment_file(input_path, output_dir, num_augmented)

# Використання:
augment_audio_folder("./dataset/custom_16k/wind", "./dataset/custom_16k/wind", num_augmented=4)
