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

def evaluate_with_recall(output_path, asd_results_path, csv_path, neg_threshold=None,
                         masking_prob=0.0): # AP
    fname = 'active_speaker_test.csv'
    labels = ['trackid', 'duration', 'fps', 'labels', 'start_frame']
    df = pd.read_csv(f'{csv_path}/{fname}', names=labels, delimiter='\t')
    output_path = os.path.join(output_path, str(masking_prob))
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


def evaluate(output_path, masking_prob=0.0):
    # calculate mAP of just output path, identity agnostically
    hyp = []
    ref = []
    output_path = os.path.join(output_path, str(masking_prob))
    print('total amount of output files: ', len(glob(f'{output_path}/*.json')))
    for trackid in tqdm(glob(f'{output_path}/*.json')):
        with open(trackid, 'r') as f:
            result = json.load(f)
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


def main(config: Dict):
    """Main training workflow"""
    
    # Load data
    splits, uttid2clipid, clip_id2spk_id2img_paths = annot_loaders.load_annotations(config['annotation_path'])
    name2voice_emb, name2face_emb = annot_loaders.load_embeddings(config['data_path'])
    clip_ids = set([utt_id.split('_clipped')[0] for utt_id in splits['test']])
    utt_id2concurrent_tracks = annot_loaders.load_manifest(clip_ids, config['annotation_path'], config['bbox_path'], config['frame_rate'],
                                                           masking_prob=config['masking_prob']) 
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
    inferer.run_inference(utt_id2concurrent_tracks, masking_prob=config['masking_prob'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SL FVA ASD Inference")
    # data-loader
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--frame_rate', type=int, default=30, help='Frame rate of the video data')
    parser.add_argument('--masking_prob', type=float, default=0.0)
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
    evaluate_with_recall(config['output_path'], 
                         asd_results_path=config['bbox_path'], 
                         csv_path = config['annotPath'], 
                         neg_threshold=0.1,
                         masking_prob=config['masking_prob'])
    evaluate(config['output_path'], 
             masking_prob=config['masking_prob'])