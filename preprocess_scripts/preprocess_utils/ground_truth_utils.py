import json
from pathlib import Path

def format_gt(gt):
    formatted = []
    for spk_id, utterances in gt.items():
        for start, end in utterances:
            formatted.append({'start': start, 'end': end, 'speaker_id': spk_id})
    return formatted

def load_annotations(annot_path):
    with open(annot_path / 'av_val.json') as f:
        annot_val = json.load(f)['videos']
    with open(annot_path / 'av_train.json') as f:
        annot_train = json.load(f)['videos']
    
    val_uids = {v['video_uid'] for v in annot_val}
    train_uids = {v['video_uid'] for v in annot_train}
    assert not val_uids & train_uids, "Overlapping video IDs between splits"
    
    clip2split = {}
    for video in annot_val + annot_train:
        split = 'val' if video['video_uid'] in val_uids else 'train'
        for clip in video['clips']:
            clip2split[clip['clip_uid']] = (video['video_uid'], split)
    return annot_val + annot_train, clip2split