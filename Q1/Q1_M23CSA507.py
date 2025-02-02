"""
File: Q1_M23CSA507.py
Author: Anchit Mulye
Description: This code runs whisper model locally on the dataset.
"""

import subprocess
import time
import jiwer
import pandas as pd

dataset_path = "/Users/anchitmulye/Documents/IITJ/CodingIIT/SU/SU-2025/Q1/LJSpeech-1.1/wavs/"
metadata_file = "/Users/anchitmulye/Documents/IITJ/CodingIIT/SU/SU-2025/Q1/LJSpeech-1.1/metadata.csv"
metadata = pd.read_csv(metadata_file, delimiter="|", header=None, names=["file_name", "text", "normalized_text"])

wer_list = list()
start_time = time.time()

for index, row in metadata.iloc[:1000].iterrows():
    file_name = row["file_name"]
    text = row["text"]

    command = f"whisper --output_dir out --output_format txt --task transcribe {dataset_path}{file_name}.wav"
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(f"Executed: {command}")

        transcript_file = f"out/{file_name}.txt"
        with open(transcript_file, "r", encoding="utf-8") as ref_file:
            reference = ref_file.read().strip()

        wer = jiwer.wer(reference, text)
        wer_list.append(wer)

        print(f"Word Error Rate (WER): {wer:.4f}")
    except Exception as e:
        print(f"Error executing {command}: {e}")

metadata.loc[:999, "WER"] = wer_list

output_file = "LJSpeechOutput.csv"
metadata.to_csv(output_file, index=False, sep="|")
end_time = time.time()
execution_time = end_time - start_time

print(metadata.head())
print(f"Whisper average WER: {sum(wer_list) / len(wer_list)}")
print(f"Total execution time: {execution_time:.4f} seconds")
