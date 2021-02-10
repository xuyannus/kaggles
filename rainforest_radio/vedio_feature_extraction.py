import numpy as np
import pandas as pd
import librosa
from skimage.transform import resize
from PIL import Image

fft = 2048
hop = 512
UNIT_SR = 48000
UNIT_LENGTH = 10 * UNIT_SR

DATA_PATH = "/Users/yanxu/Downloads/rfcx-species-audio-detection"


def get_standard_period(t_min, t_max, max_len):
    center = np.round((t_min + t_max) / 2)
    beginning = max(center - UNIT_LENGTH / 2, 0)
    ending = min(beginning + UNIT_LENGTH, max_len)
    return int(beginning), int(ending)


def pre_process_train_images():
    train_tp = pd.read_csv(DATA_PATH + '/train_tp.csv')

    fmin = int(train_tp['f_max'].max() * 1.1)
    fmax = int(train_tp['f_min'].min() * 0.9)
    print('Minimum frequency: ' + str(fmin) + ', maximum frequency: ' + str(fmax))

    for index, row in train_tp.iterrows():
        wav, sr = librosa.load(f"{DATA_PATH}/train/{row['recording_id']}.flac", sr=UNIT_SR)

        t_min = float(row['t_min']) * sr
        t_max = float(row['t_max']) * sr
        beginning, ending = get_standard_period(t_min, t_max, len(wav))
        slice = wav[beginning:ending]

        mel_spec = librosa.feature.melspectrogram(slice, n_fft=fft, hop_length=hop, sr=sr, fmin=fmin, fmax=fmax, power=1.5)
        mel_spec = resize(mel_spec, (224, 400))

        mel_spec = mel_spec - np.min(mel_spec)
        mel_spec = mel_spec / np.max(mel_spec)

        # And this 0...255 is for the saving in bmp format
        mel_spec = mel_spec * 255
        mel_spec = np.round(mel_spec)
        mel_spec = mel_spec.astype('uint8')
        mel_spec = np.asarray(mel_spec)

        bmp = Image.fromarray(mel_spec, 'L')
        bmp.save(f"{DATA_PATH}/working/{row['recording_id']}_{row['species_id']}.bmp")

        if index % 100 == 99:
            print('Processed ' + str(index) + ' train examples from ' + str(len(train_tp)))


if __name__ == "__main__":
    pre_process_train_images()
