import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
CHANNELS = 2  # 또는 2로 변경해서 테스트
DTYPE = np.int16

# 간단한 사인파 데이터 생성 (1초, 440Hz)
duration = 1.0
frequency = 440.0
t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
audio_data = 0.5 * np.sin(2 * np.pi * frequency * t).astype(DTYPE)

try:
    sd.play(audio_data, samplerate=SAMPLE_RATE, blocking=True)
    sd.wait()
    print("Minimal audio playback successful")
except Exception as e:
    print(f"Minimal playback error: {e}")