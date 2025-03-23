# adapted from https://github.com/my-yy/sl_icmr2022/blob/main/scripts/2_exract_voice_emb.py

import os
import yaml
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import torch
from extractor_utils.pickle_util import save_pickle
from extractor_utils import model_util
import extractor_utils.voice_loader as voice_loader
import numpy

# Load config
CONFIG_PATH = Path(__file__).parent.parent / "configs/voice_extract.yml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)['voice_extract']

def generate_emb_dict(wav_list):
    loader = voice_loader.get_loader(
        config['n_dataloader_thread'],
        config['batch_size'],
        wav_list
    )
    the_dict = {}
    for data, lens, keys in tqdm(loader):
        try:
            core_step(data, lens, model, keys, the_dict)
        except Exception as e:
            print(keys)
            print("error:", e)
            continue
    return the_dict

def core_step(wavs, lens, model, keys, the_dict):
    with torch.no_grad():
        feats = fun_compute_features(wavs.cuda())
        feats = fun_mean_var_norm(feats, lens)
        embedding = model(feats, lens)
        embedding_npy = embedding.detach().cpu().numpy().squeeze()
    if len(embedding_npy.shape) == 1:
        embedding_npy = embedding_npy[numpy.newaxis, :]
    for key, emb in zip(keys, embedding_npy):
        key = key.split("/")[-1]
        # remove .wav extension
        key = key.split(".")[0]
        the_dict[key] = emb


def get_ecapa_model():
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
    n_mels = 80
    channels = [1024, 1024, 1024, 1024, 3072]
    kernel_sizes = [5, 3, 3, 3, 1]
    dilations = [1, 2, 3, 4, 1]
    attention_channels = 128
    lin_neurons = 192
    model = ECAPA_TDNN(input_size=n_mels, channels=channels,
                       kernel_sizes=kernel_sizes, dilations=dilations,
                       attention_channels=attention_channels,
                       lin_neurons=lin_neurons
                       )
    return model

def get_fun_norm():
    from speechbrain.processing.features import InputNormalization
    return InputNormalization(norm_type="sentence", std_norm=False)

def get_fun_compute_features():
    from speechbrain.lobes.features import Fbank

    n_mels = 80
    left_frames = 0
    right_frames = 0
    deltas = False
    compute_features = Fbank(n_mels=n_mels, left_frames=left_frames, right_frames=right_frames, deltas=deltas)
    return compute_features

if __name__ == "__main__":
    # 1. Load model
    model = get_ecapa_model().cuda()
    model_util.load_model(config['model_path'], model)
    model.eval()
    fun_compute_features = get_fun_compute_features().cuda()
    fun_mean_var_norm = get_fun_norm().cuda()

    # 2. Get audio files
    seg_type = config['manifest_file'].split('/')[-3] # type of frontend segmentation method used
    df = pd.read_csv(config['manifest_file'])
    wav_list = df['fpath'].tolist()
    
    # 3. Process and save
    the_dict = generate_emb_dict(wav_list)
    full_path = os.path.join(config['save_path'], seg_type)
    os.makedirs(full_path, exist_ok=True)
    save_pickle(os.path.join(full_path, f'voice_input.pkl'), the_dict)