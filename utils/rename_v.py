import os

valid_dirs = [
    "dataset/to_valid/self",
    "dataset/to_valid/shahed",
    "dataset/to_valid/siren",
    "dataset/to_valid/speech",
    "dataset/to_valid/wind",
    "dataset/to_valid/explosion"
]

for dir_path in valid_dirs:
    for fname in os.listdir(dir_path):
        if fname.endswith(".wav") and "v" not in fname[:-4]:
            old_path = os.path.join(dir_path, fname)
            # Формуємо нове ім'я, додаючи 'v' перед .wav
            name_part = fname[:-4]
            new_name = f"{name_part}v.wav"
            new_path = os.path.join(dir_path, new_name)
            os.rename(old_path, new_path)

print("Додавання 'v' до імен файлів у valid-каталогах завершено.")
