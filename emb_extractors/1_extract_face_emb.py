# adapted from https://github.com/my-yy/sl_icmr2022/blob/main/scripts/1_extract_face_emb.py
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True  # This line must come before other imports

# Rest of your original code
import os
import yaml
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from extractor_utils.img_loader import Dataset
from extractor_utils.pickle_util import save_pickle
from models import incep
import pandas as pd
import glob
import numpy as np

# Load config
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs/face_extract.yml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)['face_extract']

the_dict = {}
def handle_emb_batch(all_data, batch_emb, indexies):
    batch_emb = batch_emb.detach().cpu().numpy().squeeze()
    assert len(batch_emb.shape) == 2
    indexies = indexies.detach().cpu().numpy().tolist()
    for idx, emb in zip(indexies, batch_emb):
        filepath = all_data[idx]
        # if the embedding contains nan or inf print filepath
        if not (np.isfinite(emb).all()):
            print(filepath)
            print(emb)
            raise ValueError("Embedding contains NaN or Inf values.")
        if np.isnan(emb).any():
            print(filepath)
            print(emb)
            raise ValueError("Embedding contains NaN values.")
        the_dict[filepath] = emb
        
def collate_img_files(videoid2trackid, data_path):
    image_paths = []
    for video_id in list(videoid2trackid.keys()):
        video_path = os.path.join(data_path, video_id)
        for track_path in glob.glob(video_path + "/*"):
            all_jpgs = glob.glob(track_path + "/*.jpg")
            for jpg in all_jpgs:
                image_paths.append(jpg)
    return image_paths

def fun(num_workers, all_img_data, batch_size):
    the_iter = DataLoader(Dataset(all_img_data), num_workers=num_workers, batch_size=batch_size, shuffle=False,
                          pin_memory=True)
    all_data = the_iter.dataset.all_image_files
    with torch.no_grad():
        for image_tensor, indexies in tqdm(the_iter):
            emb_vec = model(image_tensor.cuda())
            handle_emb_batch(all_data, emb_vec, indexies)

if __name__ == '__main__':
    # 1. Load model
    model = incep.InceptionResnetV1(
        pretrained=config['model_settings']['pretrained'],
        classify=config['model_settings']['classify']
    )
    model.cuda()
    model.eval()

    # 2. Load manifest
    seg_type = config['manifest_file'].split('/')[-3] # type of frontend segmentation method used
    df = pd.read_csv(config['manifest_file'])
    all_jpgs = df['path'].tolist()

    # 3. Process
    fun(config['n_dataloader_thread'], all_jpgs, config['batch_size'])

    # 4. Save
    full_path = os.path.join(config['save_path'], seg_type)
    os.makedirs(full_path, exist_ok=True)
    save_pickle(os.path.join(full_path, 'face_input.pkl'), the_dict)