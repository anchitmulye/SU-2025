import os
import pandas as pd
import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class Windowing:
    def __init__(self, n_fft):
        self.n_fft = n_fft

    def hann(self):
        return 0.5 * (1 - np.cos(2 * np.pi * np.arange(self.n_fft) / (self.n_fft - 1)))

    def hamming(self):
        return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(self.n_fft) / (self.n_fft - 1))

    def rectangular(self):
        return np.ones(self.n_fft)


class UrbanSoundDataset(Dataset):
    def __init__(self, root_dir, window_type='hann', target_sample_rate=22050,
                 n_fft=2048, hop_length=512, folds=None, max_length=500):
        self.root_dir = root_dir
        self.window_type = window_type
        self.target_sample_rate = target_sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_length = max_length
        self.original_lengths = []
        self.window_tech = Windowing(self.n_fft)

        csv_path = os.path.join(root_dir, "metadata", "UrbanSound8k.csv")
        self.metadata = pd.read_csv(csv_path)
        if folds is not None:
            self.metadata = self.metadata[self.metadata['fold'].isin(folds)]

        self.file_list = [
            (os.path.join(root_dir, "audio", f"fold{row['fold']}", row['slice_file_name']), row['classID'])
            for _, row in self.metadata.iterrows()
        ]

        if window_type == 'hann':
            # self.window = torch.hann_window(n_fft)
            self.window = self.window_tech.hann()
        elif window_type == 'hamming':
            # self.window = torch.hamming_window(n_fft)
            self.window = self.window_tech.hamming()
        elif window_type == 'rect':
            # self.window = torch.ones(n_fft)
            self.window = self.window_tech.rectangular()
        else:
            raise ValueError(f"Invalid window type: {window_type}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, label = self.file_list[idx]
        waveform, sr = torchaudio.load(path)

        if sr != self.target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sample_rate)
        waveform = waveform.mean(dim=0)

        stft = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length,
                          window=self.window, return_complex=True)
        spectrogram = torch.log(torch.abs(stft) + 1e-10)
        original_length = spectrogram.shape[1]
        spectrogram = self._pad_or_truncate(spectrogram)
        return spectrogram, label, original_length

    def _pad_or_truncate(self, spec):
        if spec.shape[1] > self.max_length:
            spec = spec[:, :self.max_length]
        else:
            pad_amount = self.max_length - spec.shape[1]
            spec = torch.nn.functional.pad(spec, (0, pad_amount))
        return spec


class SpectrogramVisualizer:
    @staticmethod
    def plot_comparison(root_dir, n_fft=2048, hop_length=512):
        dataset = UrbanSoundDataset(root_dir, folds=[1])
        waveform, _ = torchaudio.load(dataset.file_list[0][0])
        waveform = torchaudio.functional.resample(waveform.mean(dim=0), 44100, 22050)

        plt.figure(figsize=(12, 8))
        windows = {
            'Hann': torch.hann_window(n_fft),
            'Hamming': torch.hamming_window(n_fft),
            'Rectangular': torch.ones(n_fft)
        }

        for i, (name, window) in enumerate(windows.items(), 1):
            stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length,
                              window=window, return_complex=True)
            spectrogram = torch.log(torch.abs(stft) + 1e-10)

            plt.subplot(3, 1, i)
            plt.imshow(spectrogram.T, aspect='auto', origin='lower', cmap='viridis')
            plt.title(f"{name} Window Spectrogram")
            plt.colorbar()
        plt.tight_layout()
        plt.show()


class AudioClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        x = x.permute(0, 2, 1)
        lengths, sort_idx = torch.sort(lengths, descending=True)
        x = x[sort_idx]

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        _, (hidden, _) = self.lstm(packed)
        return self.fc(hidden[-1])


class Trainer:
    def __init__(self, model, train_loader, test_loader, lr=0.001):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for inputs, labels, lengths in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs, lengths)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels, lengths in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs, lengths)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    def run(self, epochs=10):
        for epoch in range(epochs):
            loss = self.train_epoch()
            acc = self.evaluate()
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        return acc


def analyze_spectrogram_lengths(root_dir):
    dataset = UrbanSoundDataset(root_dir, folds=list(range(1,11)), max_length=500)
    lengths = [spec.shape[1] for spec, _ in dataset]
    print(f"Max length: {max(lengths)}")
    print(f"Average length: {sum(lengths)/len(lengths)}")
    plt.hist(lengths, bins=50)
    plt.title("Spectrogram Length Distribution")
    plt.show()


def collate_fn(batch):
    specs, labels, lengths = zip(*batch)
    specs = torch.stack(specs)
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)
    return specs, labels, lengths


if __name__ == "__main__":
    root_dir = "/Users/anchitmulye/Documents/IITJ/CodingIIT/SU/SU-2025/Q2/UrbanSound8k"

    # analyze_spectrogram_lengths(root_dir)
    max_length = 500
    SpectrogramVisualizer.plot_comparison(root_dir)

    results = {}
    for window in ['hann', 'hamming', 'rect']:
        print(f"\nTraining with {window} window...")

        # Create datasets
        train_set = UrbanSoundDataset(root_dir, window_type=window, folds=list(range(1, 10)))
        test_set = UrbanSoundDataset(root_dir, window_type=window, folds=[10])

        # Create data loaders
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(train_set, batch_size=32, shuffle=False, collate_fn=collate_fn)

        # Initialize model and trainer
        input_size = train_set[0][0].shape[0]
        model = AudioClassifier(input_size)
        trainer = Trainer(model, train_loader, test_loader)

        accuracy = trainer.run(epochs=15)
        results[window] = accuracy

    # Print final results
    print("\nFinal Results:")
    for window, acc in results.items():
        print(f"{window:10} Window: {acc:.4f} Accuracy")
