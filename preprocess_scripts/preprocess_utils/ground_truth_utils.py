import json
from pathlib import Path
import pandas as pd
from collections import defaultdict

def format_gt(gt):
    formatted = []
    for spk_id, utterances in gt.items():
        for start, end in utterances:
            formatted.append({'start': start, 'end': end, 'speaker_id': spk_id})
    return formatted


def process_annotations(df, max_gap=0.1): # this function is for formatting AVA annotations only
    """
    Process DataFrame to identify speaking periods with temporal continuity,
    handling multiple instance_ids within the same utterance.
    
    Args:
        df: Input DataFrame containing annotation data
        max_gap: Maximum allowed gap (in seconds) between consecutive
                 frames to consider them part of the same utterance
                 
    Returns:
        Dictionary of speaking periods in format {clipid: [{
            'start_time', 'end_time', 'track'
        }]}
    """
    # Filter and sort
    speaking_df = df[df['label'] == 'SPEAKING_AUDIBLE'].copy()
    speaking_df = speaking_df.sort_values(['video_id', 'frame_timestamp'])
    
    # Create time difference between consecutive frames
    speaking_df['time_diff'] = speaking_df.groupby('video_id')['frame_timestamp'].diff()
    
    # Identify group boundaries where gap exceeds max_gap or video_id changes
    speaking_df['new_group'] = (
        (speaking_df['time_diff'] > max_gap) |
        (speaking_df['video_id'] != speaking_df['video_id'].shift())
    )
    speaking_df['group_id'] = speaking_df['new_group'].cumsum()
    
    # Aggregate groups
    grouped = speaking_df.groupby(['video_id', 'group_id']).agg({
        'frame_timestamp': ['min', 'max'],
        'instance_id': lambda x: list(x.unique())
    })
    
    # Convert to target format
    result = defaultdict(list)
    for (video_id, _), group in grouped.iterrows():
        result[video_id].append({
            'start_time': float(group[('frame_timestamp', 'min')]),
            'end_time': float(group[('frame_timestamp', 'max')]),
            'track': group[('instance_id', '<lambda>')]
        })
    
    return dict(result)



def load_annotations(annot_path, is_ava=False):
    if is_ava:
        # open annot_path/val_orig.csv with pandas
        annot_val = pd.read_csv(annot_path / 'val_orig.csv')
        annot_train = pd.read_csv(annot_path / 'train_orig.csv')
        val_uids = set(annot_val['video_id'].unique())
        train_uids = set(annot_train['video_id'].unique())

        assert not val_uids & train_uids, "Overlapping video IDs between splits"

        # create clip2split
        clip2split = {}
        for video_id in val_uids:
            clip2split[video_id] = 'val'
        for video_id in train_uids:
            clip2split[video_id] = 'train'

        # process annotation
        annot_val = process_annotations(annot_val)
        annot_train = process_annotations(annot_train)
        annot = {**annot_val, **annot_train}
        return annot, clip2split
    else:
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