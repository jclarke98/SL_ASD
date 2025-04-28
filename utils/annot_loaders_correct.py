import json
import os
import sys
import pickle
import pandas as pd
from glob import glob

def load_annotations(annotation_path):
    """Load all required annotation files from the specified directory."""
    try:
        # Load dataset splits
        with open(os.path.join(annotation_path, 'splits.json'), 'r') as f:
            splits = json.load(f)
            
        # Load utterance to clip mapping
        with open(os.path.join(annotation_path, 'uttid2clipid.json'), 'r') as f:
            uttid2clipid = json.load(f)
            
        # Load clip to speaker image mapping
        with open(os.path.join(annotation_path, 'clip_id2spk_id2img_paths.json'), 'r') as f:
            clip_id2spk_id2img_paths = json.load(f)
            
        return splits, uttid2clipid, clip_id2spk_id2img_paths
    
    except FileNotFoundError as e:
        print(f"Error loading annotation files: {e}")
        sys.exit(1)

def load_manifest(clip_ids, annotation_path, bbox_path, fps = 30):
    """
    Load audio manifests from the specified directory and the bbox data 
    to return a dictionary of meta data regarding trackids concurrent to each utterance.

    Args:
        clip_ids (set): Set of clip ids.
        annotation_path (str): Path to the annotation directory.
        bbox_path (str): Path to the bbox directory.
        fps (int): Frame rate of the video.
    Returns:
        utt_id2concurrent_tracks (dict): Dictionary of meta data for each utterance.
    """
    all_bboxes_files = glob(f'{bbox_path}/*')
    # filter out bbox files that are not in the clip_ids
    all_bboxes_files = [f for f in all_bboxes_files if os.path.basename(f).split(':')[0] in clip_ids] # ensure utt_ids is required in clip_ids
    print(f"Found {len(all_bboxes_files)} bbox files.")
    # create clip wise dictionary of bbox data
    clip_id2bbox_data = {}
    for f in all_bboxes_files:
        trackid = os.path.splitext(os.path.basename(f))[0]  # Handles dots in filenames
        with open(f, 'r') as f:
            bbox_data = json.load(f) #[{"x": 1101.67, "y": 363.6, "width": 78.34,
                                    #  "height": 91.09, "frame": 3036, "video_frame": 21426, 
                                    # "clip_frame": null, "pid": "4", "activity": 0}, 
                                    # {"x": 1212.8, "y": 361.78, "width": 60.12, 
                                    # "height": 87.45, "frame": 3037, "video_frame": 21427, 
                                    # "clip_frame": null, "pid": "4", "activity": 0}]]
        clip_id = trackid.split(':')[0]
        if clip_id not in clip_id2bbox_data:
            clip_id2bbox_data[clip_id] = []
        clip_id2bbox_data[clip_id].append((trackid, bbox_data))

    try:
        # Load audio manifest
        audio_manifest = pd.read_csv(os.path.join(annotation_path, 'audio_manifest.csv')) # utt_id,clip_id,spk_id,fpath
                                                                                          
         
    except FileNotFoundError as e:
        print(f"Error loading manifest files: {e}")
        sys.exit(1)


    utt_id2concurrent_tracks = {}
    for i, row in audio_manifest.iterrows():
        utt_id = row['utt_id']
        meta = row['utt_id'].split('clipped-')[-1]
        clip_id = row['clip_id']
        gt_pid = row['spk_id']
        start_end = meta.split('_id-')[0]  # "12_3:45_6"
        start_str, end_str = start_end.split(':')  # ["12_3", "45_6"]
        start_time = float(start_str.replace('_', '.'))  # 12.3
        end_time = float(end_str.replace('_', '.'))     # 45.6
        if clip_id not in clip_ids:
            continue
        
        # get all frames of each track that are concurrent with this utterance
        concurrent_tracks = {}
        bbox_data = clip_id2bbox_data[clip_id]
        for trackid, data in bbox_data:
            # Check if any frame in this track is concurrent with the utterance
            has_concurrent = any(
                (bbox['frame'] / fps >= start_time) and (bbox['frame'] / fps <= end_time)
                for bbox in data
            )
            if not has_concurrent:
                continue  # Skip tracks with no concurrent frames
            
            # Process all frames in the track, marking each frame's concurrency
            concurrent_tracks[trackid] = []
            for bbox in data:
                frame = bbox['frame']
                time = frame / fps
                concurrent = start_time <= time <= end_time
                concurrent_tracks[trackid].append({
                    'frame': frame,
                    'activity': bbox['activity'],
                    'pid': bbox['pid'],
                    'concurrent': concurrent
                })
            assert len(concurrent_tracks[trackid]) == len(data), f"Mismatch in concurrent tracks for {utt_id}: {len(concurrent_tracks)} vs {len(data)}"
        utt_id2concurrent_tracks[utt_id] = concurrent_tracks
    return utt_id2concurrent_tracks

# def load_manifest(clip_ids, annotation_path, bbox_path, fps = 30):
#     """
#     Load audio manifests from the specified directory and the bbox data 
#     to return a dictionary of meta data regarding trackids concurrent to each utterance.

#     Args:
#         clip_ids (set): Set of clip ids.
#         annotation_path (str): Path to the annotation directory.
#         bbox_path (str): Path to the bbox directory.
#         fps (int): Frame rate of the video.
#     Returns:
#         utt_id2concurrent_tracks (dict): Dictionary of meta data for each utterance.
#     """
#     all_bboxes_files = glob(f'{bbox_path}/*')
#     # filter out bbox files that are not in the clip_ids
#     all_bboxes_files = [f for f in all_bboxes_files if os.path.basename(f).split(':')[0] in clip_ids] # ensure utt_ids is required in clip_ids
#     print(f"Found {len(all_bboxes_files)} bbox files.")
#     # create clip wise dictionary of bbox data
#     clip_id2bbox_data = {}
#     for f in all_bboxes_files:
#         trackid = os.path.splitext(os.path.basename(f))[0]  # Handles dots in filenames
#         with open(f, 'r') as f:
#             bbox_data = json.load(f) #[{"x": 1101.67, "y": 363.6, "width": 78.34,
#                                     #  "height": 91.09, "frame": 3036, "video_frame": 21426, 
#                                     # "clip_frame": null, "pid": "4", "activity": 0}, 
#                                     # {"x": 1212.8, "y": 361.78, "width": 60.12, 
#                                     # "height": 87.45, "frame": 3037, "video_frame": 21427, 
#                                     # "clip_frame": null, "pid": "4", "activity": 0}]]
#         clip_id = trackid.split(':')[0]
#         if clip_id not in clip_id2bbox_data:
#             clip_id2bbox_data[clip_id] = []
#         clip_id2bbox_data[clip_id].append((trackid, bbox_data))

#     try:
#         # Load audio manifest
#         audio_manifest = pd.read_csv(os.path.join(annotation_path, 'audio_manifest.csv')) # utt_id,clip_id,spk_id,fpath
                                                                                          
         
#     except FileNotFoundError as e:
#         print(f"Error loading manifest files: {e}")
#         sys.exit(1)


#     utt_id2concurrent_tracks = {}
#     for i, row in audio_manifest.iterrows():
#         utt_id = row['utt_id']
#         meta = row['utt_id'].split('clipped-')[-1]
#         clip_id = row['clip_id']
#         gt_pid = row['spk_id']
#         start_end = meta.split('_id-')[0]  # "12_3:45_6"
#         start_str, end_str = start_end.split(':')  # ["12_3", "45_6"]
#         start_time = float(start_str.replace('_', '.'))  # 12.3
#         end_time = float(end_str.replace('_', '.'))     # 45.6
#         if clip_id not in clip_ids:
#             continue
        
#         # get all frames of each track that are concurrent with this utterance
#         concurrent_tracks = {}
#         bbox_data = clip_id2bbox_data[clip_id]
#         for trackid, data in bbox_data:
#             is_concurrent = False
#             for bbox in data:
#                 frame = bbox['frame']
#                 pid = bbox['pid']
#                 if frame/fps >= float(start_time) and frame/fps <= float(end_time):
#                     is_concurrent = True
#                     if trackid not in concurrent_tracks:
#                         concurrent_tracks[(trackid)] = []
#                     concurrent_tracks[trackid].append({'frame': frame, 'activity': bbox['activity'], 'pid': pid, 'concurrent': True})
#                 elif is_concurrent:
#                     if trackid not in concurrent_tracks:
#                         concurrent_tracks[(trackid)] = []
#                     concurrent_tracks[trackid].append({'frame': frame, 'activity': bbox['activity'], 'pid': pid, 'concurrent': False})
#         utt_id2concurrent_tracks[utt_id] = concurrent_tracks
#     return utt_id2concurrent_tracks

def load_embeddings(data_path):
    """Load voice and face embeddings from pickle files."""
    try:
        # Load voice embeddings
        with open(os.path.join(data_path, 'voice_input.pkl'), 'rb') as f:
            name2voice_emb = pickle.load(f)
            
        # Load face embeddings
        with open(os.path.join(data_path, 'face_input.pkl'), 'rb') as f:
            name2face_emb = pickle.load(f)
            
        return name2voice_emb, name2face_emb
    
    except FileNotFoundError as e:
        print(f"Error loading embedding files: {e}")
        sys.exit(1)