import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from .face_cropper import load_schema, crop_and_save_images

def create_uttid2clipid(uttid_map, save_path):
    os.makedirs(save_path, exist_ok=True)
    mapping = {utt_id: utt_id.split('_')[0] for utt_id in uttid_map}
    with open(Path(save_path)/'uttid2clipid.json', 'w') as f:
        json.dump(mapping, f, indent=4)
    return mapping

def create_audio_manifest(uttid_map, save_path):
    os.makedirs(save_path, exist_ok=True)
    manifest = [[uid, uid.split('_')[0], info['speaker_id'], info['file_path']]
               for uid, info in uttid_map.items()]
    pd.DataFrame(manifest, columns=['utt_id','clip_id','spk_id','fpath']
                ).to_csv(Path(save_path)/'audio_manifest.csv', index=False)

def create_data_splits(utt_parent_map, save_path, val_ratio):
    test_vids = {vid for uid, (vid, split) in utt_parent_map.items() if split == 'val'}
    train_vids = {vid for uid, (vid, split) in utt_parent_map.items() if split == 'train'}
    val_vids = set(np.random.choice(list(train_vids), int(len(train_vids)*val_ratio), False))
    
    splits = {'train': [], 'val': [], 'test': []}
    for uid, (vid, _) in utt_parent_map.items():
        if vid in (train_vids - val_vids):
            splits['train'].append(uid)
        elif vid in val_vids:
            splits['val'].append(uid)
        elif vid in test_vids:
            splits['test'].append(uid)
    
    with open(Path(save_path)/'splits.json', 'w') as f:
        json.dump(splits, f, indent=4)
    return splits


def create_image_manifest(clip_id2spk_id2img_paths, save_path):
    save_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"clip_id": cid, "spk_id": pid, "path": str(p)}
        for cid, spk_dict in clip_id2spk_id2img_paths.items()
        for pid, paths in spk_dict.items()
        for p in paths
    ]).to_csv(save_path / 'image_manifest.csv', index=False)

def save_clip_id2spk_id2img_paths(clip_id2spk_id2img_paths, save_path):
    with open(save_path / 'clip_id2spk_id2img_paths.json', 'w') as f:
        json.dump(clip_id2spk_id2img_paths, f, indent=4)
    return clip_id2spk_id2img_paths


def process_image_data(config):
    """Handle image processing pipeline."""
    df_train = load_schema(config.annot_path / 'csv', 'train')
    df_val = load_schema(config.annot_path / 'csv', 'val')
    full_df = pd.concat([df_train, df_val], ignore_index=True)

    clip_id2spk_id2img_paths = crop_and_save_images(
        df=full_df,
        bbox_path=config.annot_path / 'bbox',
        img_path=config.img_path,
        save_path=config.output_direc
    )
    create_image_manifest(clip_id2spk_id2img_paths, config.output_direc / 'annot')
    save_clip_id2spk_id2img_paths(clip_id2spk_id2img_paths, config.output_direc / 'annot')
