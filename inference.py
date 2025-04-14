"""
Self-lifting for audiovisual active speaker detection

This script performs inference using a trained checkpoint for SL-ASD model on the Ego4D dataset.
It saves the probability of identity correspondence for frames within tracks that are
concurrent to a given utterance. It saves these probabilities in the standard .json format, 
consistent with Ego4D. Then it evalutes via AP and recall.

evaluate_with_recall evalutes for utterances in the dataset (including those missed by the
utterance segmentation front-end), and also evaluates the performance of the front-end segmentation
method in terms of recall.

evaluate just evalutes by theAP of the output path, identity agnostically.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import argparse
from typing import Dict
import torch
from tqdm import tqdm
import json
import time
from glob import glob
from sklearn.metrics import average_precision_score
import pandas as pd
import yaml
from pathlib import Path

from dataloader import get_dataloader
from models.sl_ASD_model import Encoder, Decoder
from utils import model_util, annot_loaders
from sl_ASD import Model

from collections import defaultdict
import matplotlib.pyplot as plt

def evaluate_with_recall(output_path, asd_results_path, csv_path, neg_threshold=None): # AP
    fname = 'active_speaker_test.csv'
    labels = ['trackid', 'duration', 'fps', 'labels', 'start_frame']
    df = pd.read_csv(f'{csv_path}/{fname}', names=labels, delimiter='\t')

    hyp = []
    ref = []
    n = 0
    label_length = [] # list of length of missed labels
    # iterate through the trackids
    for trackid in tqdm(df['trackid']):
        with open(f'{asd_results_path}/{trackid}.json', 'r') as f:
            result = json.load(f)
        gt_length = len(result)
        # check if trackid exists in output_path
        if os.path.exists(f'{output_path}/{trackid}.json'):
            with open(f'{output_path}/{trackid}.json', 'r') as f:
                result = json.load(f)

                assert gt_length == len(result), f'length of gt and result do not match for {trackid}'
            # n += 1
            for frame in result:
                score = float(frame['score'])
                label = int(frame['activity'])
                hyp.append(score)
                ref.append(label)
        # get from asd_results_path
        else:
            with open(f'{asd_results_path}/{trackid}.json', 'r') as f:
                result = json.load(f)
            labels = []
            for frame in result:
                # user 'activity or 'label' depending on the json file
                try:
                    label = int(frame['activity'])
                except KeyError:
                    label = int(frame['label'])
                labels.append(label)
            if int(1) in labels:
                label_length.append(len([l for l in labels if l == 1]))
                # print(trackid, 'has speech', ', here are the labels: ', labels)
                n+=1
            for frame in result:
                score = 0
                # score = float(neg_threshold)
                label = int(frame['activity'])    
                hyp.append(score)
                ref.append(label)
            

    print("Evaluation metric (AP):", average_precision_score(ref, hyp))
    print('proportion of tracks containing speech that were not detected by segmentation: ',
           round(100*n/len(df['trackid']),2),'%')
    print('average length of missed labels: ', sum(label_length)/len(label_length)/30, 'seconds')


def evaluate(output_path):
    # calculate mAP of just output path, identity agnostically
    hyp = []
    ref = []
    print('total amount of output files: ', len(glob(f'{output_path}/*.json')))
    for trackid in tqdm(glob(f'{output_path}/*.json')):
        with open(trackid, 'r') as f:
            result = json.load(f)
        # scrs = [1 if frame['score'] >= 0.01 else 0 for frame in result]
        scrs = [float(frame['score']) for frame in result]
        labs = [int(frame['activity']) for frame in result]
        clipid = trackid.split('/')[-1][:36]
        for frame in result:
            try:
                hyp.append(float(frame['score']))
                ref.append(int(frame['activity']))
                clipid = trackid.split('/')[-1][:36]
            except KeyError:
                print(f"Missing 'score' or 'label' in segment: {frame}")
                continue
    print("Evaluation metric (AP):", average_precision_score(ref, hyp))

    # # delelte the output_path folder
    # c_time = str(int(time.time()))
    # os.system(f'mv {output_path} /mnt/parscratch/users/acp21jrc/rubbish/{c_time}')


# def evaluate(output_path):
#     """Calculate mAP stratified by clip ID with timeline visualizations."""
#     clips_dict = defaultdict(lambda: {'hyp': [], 'ref': []})
#     json_files = glob(f'{output_path}/*.json')
#     print(f'Total amount of output files: {len(json_files)}')

#     for track_path in tqdm(json_files, desc='Processing clips'):
#         with open(track_path, 'r') as f:
#             result = json.load(f)
        
#         clip_id = os.path.basename(track_path)[:36]
#         track_id = os.path.splitext(os.path.basename(track_path))[0]
        
#         # Data collection for both evaluation and visualization
#         times, scores, activities = [], [], []
#         for i, frame in enumerate(result):
#             # Collect evaluation metrics
#             clips_dict[clip_id]['hyp'].append(float(frame['score']))
#             clips_dict[clip_id]['ref'].append(int(frame['activity']))
        
#             # Collect visualization data
#             times.append(float(i))  # Assuming 'time' field exists
#             scores.append(float(frame['score']))
#             activities.append(int(frame['activity']))
            
#         # Generate timeline visualization
#         if times:
#             plt.figure(figsize=(15, 5))
            
#             # Plot ground truth activity
#             plt.step(times, activities, where='post', 
#                     label='Ground Truth', color='#2ca02c', linewidth=2)
            
#             # Plot predicted scores
#             plt.plot(times, scores, label='Predicted Score', 
#                    color='#1f77b4', alpha=0.7, linewidth=1.5)
#             plt.ylim(-0.05, 1.1)
#             plt.xlabel('Time (seconds)', fontsize=12)
#             plt.ylabel('Activity/Score', fontsize=12)
#             plt.title(f'{track_id}\nActivity vs Prediction', fontsize=14)
#             plt.legend(loc='upper right')
#             plt.grid(alpha=0.3)
#             plt.tight_layout()

#             # Save visualization
#             vis_dir = os.path.join('visualisations', clip_id)
#             os.makedirs(vis_dir, exist_ok=True)
#             plt.savefig(os.path.join(vis_dir, f'{track_id}.png'), 
#                       dpi=150, bbox_inches='tight')
#             plt.close()

#     # Existing evaluation metrics calculation
#     ap_results = {}
#     for clip_id, data in clips_dict.items():
#         if not data['ref']:
#             continue
#         try:
#             ap = average_precision_score(data['ref'], data['hyp'])
#             ap_results[clip_id] = ap
#         except ValueError as e:
#             print(f"Error calculating AP for {clip_id}: {e}")
#             continue

#     if not ap_results:
#         print("No valid clips for evaluation")
#         return 0.0

#     mean_ap = sum(ap_results.values()) / len(ap_results)
#     print("\nClip-wise AP Scores:")
#     for clip_id, ap in ap_results.items():
#         print(f"{clip_id}: {ap:.4f}")
    
#     print(f"\nMean Average Precision (mAP): {mean_ap:.4f}")
#     # delelte the output_path folder
#     c_time = str(int(time.time()))
#     os.system(f'mv {output_path} /mnt/parscratch/users/acp21jrc/rubbish/{c_time}')
    
#     return mean_ap


    
# def evaluate(output_path):
#     # Dictionary to hold scores and labels per class (pid)
#     results_by_pid = {}

#     print('Total amount of output files:', len(glob(f'{output_path}/*.json')))
#     for trackid in tqdm(glob(f'{output_path}/*.json')):
#         with open(trackid, 'r') as f:
#             result = json.load(f)
#         clipid = trackid.split('/')[-1][:36]
#         for frame in result:
#             try:
#                 score = float(frame['score'])
#                 label = int(frame['activity'])
#                 pid = str(frame['pid'])
#                 # Create a unique class identifier by combining clipid and pid.
#                 unique_id = clipid + pid
                
#                 # If this unique_id hasn't been seen, initialize lists
#                 if unique_id not in results_by_pid:
#                     results_by_pid[unique_id] = {"scores": [], "labels": []}

#                 results_by_pid[unique_id]["scores"].append(score)
#                 results_by_pid[unique_id]["labels"].append(label)
#             except KeyError:
#                 print(f"Missing 'score' or 'activity' in segment: {frame}")
#                 continue

#     # Compute average precision for each class
#     ap_values = []
#     for unique_id, data in results_by_pid.items():
#         # Only compute AP if there is at least one positive sample
#         if sum(data["labels"]) > 0:
#             ap = average_precision_score(data["labels"], data["scores"])
#             ap_values.append(ap)
#         else:
#             # If there are no positive labels, average precision is undefined.
#             # Here we define it as 0.0 (or you could choose to skip it).
#             ap_values.append(0.0)

#     # Calculate the mean Average Precision (mAP)
#     if ap_values:
#         mean_ap = sum(ap_values) / len(ap_values)
#     else:
#         mean_ap = 0.0

#     print("Evaluation metric (mAP):", mean_ap)


def main(config: Dict):
    """Main training workflow"""
    
    # Load data
    splits, uttid2clipid, clip_id2spk_id2img_paths = annot_loaders.load_annotations(config['annotation_path'])
    name2voice_emb, name2face_emb = annot_loaders.load_embeddings(config['data_path'])
    clip_ids = set([utt_id.split('_clipped')[0] for utt_id in splits['test']])
    utt_id2concurrent_tracks = annot_loaders.load_manifest(clip_ids, config['annotation_path'], config['bbox_path'], config['frame_rate']) 
    test_loader = get_dataloader(
        utt_ids=splits['test'],
        name2voice_emb=name2voice_emb,
        name2face_emb=name2face_emb,
        uttid2clipid=uttid2clipid,
        clip_id2spk_id2img_paths=clip_id2spk_id2img_paths,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=False,
        split='inference',
        visible_only=True,
        utt_id2concurrent_tracks=utt_id2concurrent_tracks
    )

    # Initialize model components
    encoder = Encoder()
    if config['encoder_weights'] is not None and os.path.exists(config['encoder_weights']):
        model_util.load_model(
            config['encoder_weights'], 
            encoder, 
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            strict=True,
            )
    decoder = Decoder(
        128,
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        num_heads_cross=config['num_cross_heads']
    )
    
    # Initialize trainer
    inferer = Model(
        config=config,
        encoder=encoder,
        decoder=decoder,
        mode='inference',
        test_loader=test_loader,
    )
    inferer.load_checkpoint(config['checkpoint_path'])

    
    # Run inference
    inferer.run_inference(utt_id2concurrent_tracks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SL FVA ASD Inference")
    # data-loader
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--frame_rate', type=int, default=30, help='Frame rate of the video data')
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=100000)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_cross_heads', type=int, default=4)
    args = parser.parse_args()
    
    # Convert to dictionary config
    config = vars(args)
    # Load config
    CONFIG_PATH = "configs/sl_ASD.yml"
    with open(CONFIG_PATH, 'r') as f:
        paths = yaml.safe_load(f)['inference']
    config = {**config, **paths}


    # Run training
    main(config)
    evaluate_with_recall(config['output_path'], asd_results_path=config['bbox_path'], csv_path = config['annotPath'], neg_threshold=0.1)
    evaluate(config['output_path'])