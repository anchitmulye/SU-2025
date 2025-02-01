import librosa
import librosa.display
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

audio_files = {
    "Rock": "songs/happy-pop-country-village-rock-250547.mp3",
    "Pop": "songs/love-love-and-love-289967.mp3",
    "Piano": "songs/ethereal-visit-252409.mp3",
    "Dance": "songs/upbeat-background-music-212772.mp3"
}

# Parameters
sr = 22050
n_fft = 2048
hop_length = 512
n_mels = 128
spectrograms = {}
plt.figure(figsize=(10, 8))
for i, (genre, file) in enumerate(audio_files.items()):
    print(file)
    if not os.path.exists(file):
        print(f"File {file} not found. Skipping {genre}.")
        continue

    y, _ = librosa.load(file, sr=sr)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    spectrograms[genre] = torch.tensor(mel_spec_db)

    plt.subplot(2, 2, i + 1)
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"{genre} Spectrogram")

plt.tight_layout()
plt.show()


# Analysis 
def analyze_spectrograms(spectrograms):
    for genre, spec in spectrograms.items():
        print(f"\n{genre} Analysis:")
        print(f"- Frequency range: {spec.shape[0]} Mel bands")
        print(f"- Time resolution: {spec.shape[1]} frames")
        print(f"- Intensity variation: {torch.std(spec).item():.2f} dB")
        print(f"- Spectral contrast (approx): {torch.mean(spec).item():.2f} dB")


analyze_spectrograms(spectrograms)
