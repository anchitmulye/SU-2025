
# Speech-to-Text (STT) on LJSpeech1.1 and Spectrogram Analysis using UrbanSound8K

This repository contains the implementation for various Speech-to-Text (STT) models, including **Nova-2** and **Whisper**, and their evaluation on audio datasets such as the **UrbanSound8K** and **LJSpeech1.1** datasets. The project covers:

- Applying different windowing techniques for signal processing.
- Generating and analyzing spectrograms for audio data.
- Comparing the performance of STT models using Word Error Rate (WER).
- Training classifiers on features extracted from spectrograms.

## Datasets Used

1. **UrbanSound8K**:
   - This dataset contains 8,732 labeled sound excerpts from urban environments, classified into 10 different categories.
   - [Link to dataset](https://www.kaggle.com/datasets/urbansound8k/urbansound8k)
   
2. **LJSpeech1.1**:
   - This dataset contains 13,100 short audio clips of a single speaker reading passages from 7 non-fiction books.
   - [Link to dataset](https://www.kaggle.com/datasets/mathurinache/the-lj-speech-dataset)

## Windowing Techniques Implemented

Three different windowing techniques were implemented and applied on the audio signal:

1. **Hann Window**:
   - A smooth window that reduces spectral leakage, commonly used in signal processing.
2. **Hamming Window**:
   - Similar to the Hann window but with a slightly different shape to reduce side lobes.
3. **Rectangular Window**:
   - A simple window with constant amplitude, often used in basic STFT implementations.

Each window was applied to the signal to observe how they affect the resulting spectrogram and signal processing.

## Model Evaluation

We used the **Word Error Rate (WER)** metric to evaluate the performance of the **Nova-2** and **Whisper** STT models on the **UrbanSound8K** and **LJSpeech1.1** datasets. 

### WER Formula:
\[
WER = \frac{S + D + I}{N}
\]

Where:
- **S**: Number of substitutions
- **D**: Number of deletions
- **I**: Number of insertions
- **N**: Total number of words in the reference transcript


## Usage

### 1. Data Preprocessing:
- Load the audio files from the **UrbanSound8K** and **LJSpeech1.1** datasets.
- Apply windowing techniques to the audio signals.

### 2. Spectrogram Generation:
- Generate spectrograms using the **Short-Time Fourier Transform (STFT)** method.
- Visualize the spectrograms for analysis.

### 3. Model Evaluation:
- Use the **Nova-2** and **Whisper** STT models to transcribe the audio.
- Evaluate their performance using **WER**.

### 4. Classifier Training:
- Extract features from the spectrograms.
- Train classifiers (e.g., **SVM** or **Neural Network**) for audio classification.

## Results
_Results(Q1)_  
The results of this project show that the **Whisper** model outperforms **Nova-2** in terms of **Word Error Rate (WER)** on the **LJSpeech1.1** dataset. The classifier trained on spectrogram features achieved an accuracy of **X%** for the audio classification task.

_Results(Q2) TaskA_  
Accuracy is maximum with the Hann Windowing Technique

_Results(q2) TaskB_:
- **Rock** genre had the highest intensity variation, with contrasting loud and soft parts.
- **Pop** genre showed balanced spectral contrast.
- **Piano** genre had smooth sound transitions with lower intensity variation.
- **Dance** genre showed the highest intensity variation, with energetic and bass-heavy sections.

