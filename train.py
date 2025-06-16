"""
Self-lifting for audiovisual active speaker detection

This script trains the SL-ASD model on the Ego4D dataset using:
- Pretrained/finetuned frozen face-recognition branch/speaker-recognition branch encoders
- Learnable decoder with self-attention and cross-attention mechanisms
"""

import os
import argparse
from typing import Dict
import torch
import wandb
from pathlib import Path
import yaml

from dataloader import get_dataloader
from models.sl_ASD_model import Encoder, Decoder
from utils import model_util, annot_loaders
from sl_ASD import Model

def main(config: Dict):
    """Main training workflow"""
    # Initialize WandB
    if config['wandb']:
        wandb.init(project='sl_ASD', config=config)
    
    # Load data
    splits, uttid2clipid, clip_id2spk_id2img_paths = annot_loaders.load_annotations(config['annotation_path'])
    name2voice_emb, name2face_emb = annot_loaders.load_embeddings(config['data_path'])
    train_loader = get_dataloader(
        utt_ids=splits['train'],
        name2voice_emb=name2voice_emb,
        name2face_emb=name2face_emb,
        uttid2clipid=uttid2clipid,
        clip_id2spk_id2img_paths=clip_id2spk_id2img_paths,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        split='train',
        visible_only=config['visible_only']
    )
    val_loader = get_dataloader(
        utt_ids=splits['val'],
        # utt_ids=splits['test'],  # Use test split for validation
        name2voice_emb=name2voice_emb,
        name2face_emb=name2face_emb,
        uttid2clipid=uttid2clipid,
        clip_id2spk_id2img_paths=clip_id2spk_id2img_paths,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=False,
        split='val',
        visible_only=config['visible_only']
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
        dropout=config['dropout'],
        num_heads_cross=config['num_cross_heads'],
        visible_only=config['visible_only']
    )
    
    # Initialize trainer
    trainer = Model(
        config=config,
        encoder=encoder,
        decoder=decoder,
        mode='train',
        train_loader=train_loader,
        val_loader=val_loader
    )

    # print the number of learnable parameters in the decoder
    num_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Number of learnable parameters in decoder: {num_params}")
    
    # Run training
    trainer.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SL FVA ASD Training")
    # data-loader
    parser.add_argument('--num_workers', type=int, default=4)
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-4) # 5e-5
    parser.add_argument('--lr_decay', type=float, default=0.9) # 0.9
    parser.add_argument('--lr_step_size', type=int, default=2) # 2
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.35) # 35
    parser.add_argument('--num_cross_heads', type=int, default=4)
    parser.add_argument('--patience', type=int, default=20)
    # wandb
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    

    # Convert to dictionary config
    config = vars(args)
    config['visible_only'] = True # False # (True ignores -1 and 0, i.e. miscellaneous background voice, and camera-wearer voice, respectively)
    
    # Load config
    CONFIG_PATH = "configs/sl_ASD.yml"
    with open(CONFIG_PATH, 'r') as f:
        paths = yaml.safe_load(f)['train']
    config = {**config, **paths}

    # Run training
    main(config)