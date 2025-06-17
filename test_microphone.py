import sounddevice as sd
import scipy.io.wavfile as wav

duration = 10  # секунди
fs = 16000
print("Запис...")
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait()
print("Готово. Збереження...")
wav.write("test_output.wav", fs, recording)
