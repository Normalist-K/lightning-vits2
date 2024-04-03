from soundfile import read
import torch
import numpy as np



def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def load_wav_to_torch(full_path):
    data, sampling_rate = read(full_path, dtype='float32')
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate
    # data, sampling_rate = torchaudio.load(full_path)
    # return data, sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text