import os
import librosa
import soundfile as sf

def resample_and_convert_to_mono(input_dir, output_dir, target_sr=16000):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Завантаження без зміни SR
            y, sr = librosa.load(input_path, sr=None, mono=False)

            # Примусове зведення в моно
            if y.ndim > 1:
                y = librosa.to_mono(y)

            # Ресемплінг до 16 кГц
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

            # Збереження
            sf.write(output_path, y_resampled, target_sr)

# Приклад використання:
resample_and_convert_to_mono("dataset/custom_44k/noise", "dataset/custom_16k/noise")
