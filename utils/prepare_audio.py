import os
import librosa
import soundfile as sf
import numpy as np
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

random.seed(42)

INPUT_BASE = "dataset/custom_16k"
OUTPUT_BASE = "dataset"
CLASSES = ["self", "shahed", "speech", "wind", "siren", "explosion"]
FRAGMENT_DURATION = 1.0
MIN_OVERLAP = 0.15
MAX_OVERLAP = 0.25
SR = 16000

def slice_audio(y, sr, step, duration=1.0):
    samples = []
    frame_len = int(duration * sr)
    step_len = int(step * sr)
    for start in range(0, len(y) - frame_len + 1, step_len):
        samples.append(y[start:start + frame_len])

    remaining = len(y) % step_len
    if remaining >= 0.5 * sr and remaining < frame_len:
        pad_y = np.tile(y[-remaining:], int(np.ceil(frame_len / remaining)))[:frame_len]
        samples.append(pad_y)

    return samples

def process_class(class_name):
    input_dir = os.path.join(INPUT_BASE, class_name)
    output_train = os.path.join(OUTPUT_BASE, "to_train", class_name)
    output_valid = os.path.join(OUTPUT_BASE, "to_valid", class_name)
    os.makedirs(output_train, exist_ok=True)
    os.makedirs(output_valid, exist_ok=True)

    fragments = []
    for file in os.listdir(input_dir):
        if not file.lower().endswith(".wav"):
            continue
        path = os.path.join(input_dir, file)
        y, sr_file = librosa.load(path, sr=SR)
        overlap = random.uniform(MIN_OVERLAP, MAX_OVERLAP)
        step = FRAGMENT_DURATION * (1 - overlap)
        fragments += slice_audio(y, SR, step)

    train_fragments, valid_fragments = train_test_split(fragments, test_size=0.2, random_state=42)

    for idx, frag in enumerate(train_fragments):
        out_path = os.path.join(output_train, f"{class_name}_{idx:05d}.wav")
        sf.write(out_path, frag, SR)

    for idx, frag in enumerate(valid_fragments):
        out_path = os.path.join(output_valid, f"{class_name}_{idx:05d}.wav")
        sf.write(out_path, frag, SR)

for class_name in CLASSES:
    process_class(class_name)
