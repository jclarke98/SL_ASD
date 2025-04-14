# taken from https://github.com/my-yy/sl_icmr2022/

import torch
from torch.utils.data import DataLoader
import torchaudio
import numpy as np
import librosa

class Dataset(torch.utils.data.Dataset):

    def __init__(self, all_wavs):
        self.all_wavs = all_wavs

    def __len__(self):
        return len(self.all_wavs)

    def __getitem__(self, index):
        return {
            "key": self.all_wavs[index],
            # "data": torchaudio.load(self.all_wavs[index])[0]
            'data': torch.from_numpy(librosa.load(self.all_wavs[index], sr=None)[0])
        }


# def collate_fn(item_list):
#     data_list = [i['data'] for i in item_list]
#     the_lengths = np.array([i.shape[-1] for i in data_list])
#     max_len = np.max(the_lengths)
#     len_ratio = the_lengths / max_len

#     batch_size = len(item_list)
#     output = torch.zeros([batch_size, max_len])
#     for i in range(batch_size):
#         cur = data_list[i]
#         cur_len = data_list[i].shape[-1]
#         output[i, :cur_len] = cur.squeeze()

#     len_ratio = torch.FloatTensor(len_ratio)
#     keys = [i['key'] for i in item_list]
#     return output, len_ratio, keys


def collate_fn(item_list):
    max_duration = 1920000  # 5 seconds at 16kHz
    # Get audio data and lengths from the items
    data_list = [i['data'][:max_duration] for i in item_list]
    keys = [i['key'] for i in item_list]
    
    # Determine the required minimum number of samples (e.g., 3 seconds at 16kHz)
    min_samples = 4000
    
    # Repeat audio if too short
    padded_data_list = []
    for i, audio in enumerate(data_list):
        cur_len = audio.shape[-1]
        if cur_len == 0:
            # create random l2 normalised array
            print(f"Audio {keys[i]} is empty, creating random audio")
        if cur_len < min_samples:
            # Calculate how many times we need to repeat the audio to reach min_samples
            repeat_factor = -(-min_samples // cur_len)  # ceiling division
            # Tile the audio and then take exactly min_samples
            audio = (audio.repeat(repeat_factor))[:min_samples]
        padded_data_list.append(audio)
    
    # After repetition, get the new lengths (which will be at least min_samples)
    the_lengths = np.array([audio.shape[-1] for audio in padded_data_list])
    max_len = np.max(the_lengths)
    
    batch_size = len(item_list)
    # Create a batch tensor with zeros
    output = torch.zeros([batch_size, max_len])
    for i, audio in enumerate(padded_data_list):
        cur_len = audio.shape[-1]
        output[i, :cur_len] = audio.squeeze()

    len_ratio = torch.FloatTensor(the_lengths / max_len)
    return output, len_ratio, keys


def get_loader(num_workers, batch_size, all_wavs):
    loader = DataLoader(Dataset(all_wavs),
                        num_workers=num_workers, batch_size=batch_size,
                        shuffle=False, pin_memory=True, collate_fn=collate_fn)
    return loader


if __name__ == "__main__":
    pass
