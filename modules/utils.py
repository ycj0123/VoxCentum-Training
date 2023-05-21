# Third Party
import librosa
import numpy as np
import random
import torch
# ===============================================
#       code from Arsha for loading data.
# This code extract features for a give audio file
# ===============================================

## Loading and Transforms


def load_wav(audio_filepath, sr, min_dur_sec=4):
    audio_data, fs = librosa.load(audio_filepath, sr=16000)
    audio_data = librosa.util.normalize(audio_data)
    len_file = len(audio_data)

    if len_file < int(min_dur_sec*sr):
        dummy = np.zeros((1, int(min_dur_sec*sr)-len_file))
        extened_wav = np.concatenate((audio_data, dummy[0]))
    else:
        extened_wav = audio_data
    return extened_wav


def mel_spec_from_wav(wav, hop_length, win_length, n_mels):
    linear = librosa.feature.melspectrogram(
        y=wav, n_mels=n_mels, win_length=win_length, hop_length=hop_length)  # mel spectrogram
    return linear.T


def lin_spec_from_wav(wav, hop_length, win_length, n_fft=512):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)  # linear spectrogram
    # output size ≈ (1 + n_fft/2, seconds*sr/hop_length)
    return linear.T

# Dataset and Feature Extraction

# Used by WaveformDataset


def load_waveform(filepath, sr=16000, min_dur_sec=4, mode='train', wf_sec=4):
    wf = load_wav(filepath, sr=sr, min_dur_sec=min_dur_sec)
    wf_len = wf_sec * sr

    if mode == 'train':
        try:
            randtime = np.random.randint(0, wf.shape[0] - wf_len + 1)
            wf_sample = wf[randtime:randtime + wf_len]
        except:
            raise ValueError(f'The shape of wf {wf.shape[0]} <= the sample length {wf_len}.')
    else:
        wf_sample = wf
    wf_sample = torch.from_numpy(wf_sample)

    return wf_sample

# Used by SpeechDataset, deprecated


def load_data(filepath, sr=16000, mel=False, min_dur_sec=4, win_length=400, hop_length=160, n_fft=512, spec_len=400, mode='train', n_mels=128):
    assert spec_len <= min_dur_sec * sr // hop_length, "min_dur_sec must not be smaller than sample_sec!"
    audio_data = load_wav(filepath, sr=sr, min_dur_sec=min_dur_sec)
    if mel == True:
        spect = mel_spec_from_wav(audio_data, hop_length, win_length, n_mels)
    else:
        spect = lin_spec_from_wav(audio_data, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(spect)  # magnitude
    mag_T = mag.T

    if mode == 'train':
        randtime = np.random.randint(0, mag_T.shape[1]-spec_len)
        spec_mag = mag_T[:, randtime:randtime+spec_len]
    else:
        spec_mag = mag_T

    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    return (spec_mag - mu) / (std + 1e-5)

# Deprecated


def feature_extraction(filepath, sr=16000, min_dur_sec=4, win_length=400, hop_length=160, n_mels=256, spec_len=400, mode='train'):
    audio_data = load_wav(filepath, sr=sr, min_dur_sec=min_dur_sec)
    # mel_spect = mel_spec_from_wav(audio_data, hop_length, win_length, n_mels)
    lin_spect = lin_spec_from_wav(audio_data, hop_length, win_length, n_fft=512)
    mag, _ = librosa.magphase(lin_spect)  # magnitude
    mag_T = mag.T
    mu = np.mean(mag_T, 0, keepdims=True)
    std = np.std(mag_T, 0, keepdims=True)
    return (mag_T - mu) / (std + 1e-5)


# Used by SpeechFeatureDataset, deprecated
def load_npy_data(filepath, spec_len=400, mode='train'):
    mag_T = np.load(filepath)
    if mode == 'train':
        randtime = np.random.randint(0, mag_T.shape[1]-spec_len)
        spec_mag = mag_T[:, randtime:randtime+spec_len]
    else:
        spec_mag = mag_T
    return spec_mag

# Collate Functions


# def speech_collate(batch):
#     specs = []
#     targets = []
#     if len(batch[0]) == 3:
#         families = []
#     else:
#         families = None
#     for sample in batch:
#         specs.append(sample[0])
#         targets.append((sample[1]))
#         if families is not None:
#             families.append((sample[2]))
#     if families is None:
#         return specs, targets
#     else:
#         return specs, targets, families

def speech_collate(batch):
    collated = [[] for x in range(len(batch[0]))]
    for sample in batch:
        for i, s in enumerate(sample):
            collated[i].append(s)
    return collated

def fleurs_collate_pad(batch):
    wfs = []
    labels = []
    longest = max(batch, key=lambda sample: sample['audio']['array'].shape[0])['audio']['array']
    max_len = longest.shape[0] if longest.shape[0] < 480000 else 480000
    for sample in batch:
        new_sample = np.zeros(max_len, dtype=sample['audio']['array'].dtype)
        if sample['audio']['array'].shape[0] <= max_len:
            new_sample[:sample['audio']['array'].shape[0]] = sample['audio']['array']
        else:
            new_sample = sample['audio']['array'][:max_len]
        wfs.append(torch.from_numpy(new_sample))
        labels.append(torch.tensor([int(sample['lang_id'])]))
    return wfs, labels

def speech_collate_pad(batch):
    targets = []
    specs = []
    longest = max(batch, key=lambda sample: sample[0].shape[1])
    max_len = longest[0].shape[1]
    for sample in batch:
        new_sample = torch.zeros((sample[0].shape[0]), max_len, dtype=sample[0].dtype)
        new_sample[:, :sample[0].shape[1]] = sample[0]
        specs.append(new_sample)
        targets.append(sample[1])
    return specs, targets


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)