import os
import torch
import csv
import librosa
import numpy as np
import pandas as pd
from skimage.transform import resize

DATA_PATH = "/Users/yanxu/Downloads/rfcx-species-audio-detection"
fft = 2048
hop = 512

# Less rounding errors this way
UNIT_SR = 48000
UNIT_LENGTH = 10 * UNIT_SR

train_tp = pd.read_csv(DATA_PATH + '/train_tp.csv')
fmin = int(train_tp['f_max'].max() * 1.1)
fmax = int(train_tp['f_min'].min() * 0.9)

model = torch.load('best_model.pt')
model.eval()

if torch.cuda.is_available():
    model.cuda()


def load_test_file(filename):
    wav, sr = librosa.load(f"{DATA_PATH}/test/{filename}", sr=None)

    # Split for enough segments to not miss anything
    segments = len(wav) / UNIT_LENGTH
    segments = int(np.ceil(segments))

    mel_array = []
    for i in range(0, segments):
        # Last segment going from the end
        if (i + 1) * UNIT_LENGTH > len(wav):
            slice = wav[len(wav) - UNIT_LENGTH:len(wav)]
        else:
            slice = wav[i * UNIT_LENGTH:(i + 1) * UNIT_LENGTH]

        # Same mel spectrogram as before
        mel_spec = librosa.feature.melspectrogram(slice, n_fft=fft, hop_length=hop, sr=sr, fmin=fmin, fmax=fmax, power=1.5)
        mel_spec = resize(mel_spec, (224, 400))

        mel_spec = mel_spec - np.min(mel_spec)
        mel_spec = mel_spec / np.max(mel_spec)
        mel_spec = np.stack((mel_spec, mel_spec, mel_spec))
        mel_array.append(mel_spec)

    return mel_array


# Prediction loop
print('Starting prediction loop')
with open(f"{DATA_PATH}/submission.csv", 'w', newline='') as csvfile:
    submission_writer = csv.writer(csvfile, delimiter=',')
    submission_writer.writerow(
        ['recording_id', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11',
         's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23'])

    test_files = os.listdir(f"{DATA_PATH}/test/")
    print(len(test_files))

    # Every test file is split on several chunks and prediction is made for each chunk
    for i in range(0, len(test_files)):
        data = load_test_file(test_files[i])
        data = torch.tensor(data)
        data = data.float()
        if torch.cuda.is_available():
            data = data.cuda()

        output = model(data)

        # Taking max prediction from all slices per bird species
        # Usually you want Sigmoid layer here to convert output to probabilities
        # In this competition only relative ranking matters, and not the exact value of prediction, so we can use it directly
        maxed_output = torch.max(output, dim=0)[0]
        maxed_output = maxed_output.cpu().detach()

        file_id = str.split(test_files[i], '.')[0]
        write_array = [file_id]

        for out in maxed_output:
            write_array.append(out.item())

        submission_writer.writerow(write_array)

        if i % 100 == 0 and i > 0:
            print('Predicted for ' + str(i) + ' of ' + str(len(test_files) + 1) + ' files')

print('Submission generated')