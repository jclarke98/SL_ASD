
import os
from typing import Dict, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import average_precision_score, precision_recall_curve
import wandb
from loss import lossAV
import json

class Model:
    """Main training class handling model training, evaluation, and inference"""
    
    def __init__(self, config: Dict, encoder: nn.Module, decoder: nn.Module, 
                 mode: str = 'train', 
                 train_loader: torch.utils.data.DataLoader = None, 
                 val_loader: torch.utils.data.DataLoader = None, 
                 test_loader: torch.utils.data.DataLoader = None,
                 ):
        """
        Args:
            config: Training configuration dictionary
            encoder: Pretrained frozen encoder
            decoder: Learnable decoder network
            mode: Purpose of the class instantiation ('train', 'eval', or 'inference')
            train_loader: Training dataloader (default: None)
            val_loader: Validation dataloader (default: None)
            test_loader: Test dataloader (default: None)
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model setup
        self.encoder = encoder.to(self.device).eval()  # Freeze encoder
        self.decoder = decoder.to(self.device)
        
        # Configure
        if mode == 'train':
            # Optimizer and loss
            self.optimizer = torch.optim.Adam(self.decoder.parameters(), lr=config['lr'])
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size= config['lr_step_size'], 
                gamma=config['lr_decay']
            )
            if train_loader is None or val_loader is None:
                raise ValueError("Train and validation loaders are required for training mode")
            self.train_loader = train_loader
            self.val_loader = val_loader
        elif mode == 'inference':
            if test_loader is None:
                raise ValueError("Test loader is required for inference mode")
            self.test_loader = test_loader
        else:
            raise ValueError("Invalid mode. Choose from 'train' or 'inference'")
        
        # Training state
        if mode == 'train':
            self.best_bap = 0.0
            self.epoch = 0
        
        # Loss function
        self.loss_fn = lossAV()
            
    def train_epoch(self) -> float:
        """Run one training epoch, returns average loss"""
        self.decoder.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {self.epoch}")
        
        for batch in progress_bar:
            # Move data to device
            voices = batch['voices'].to(self.device)
            faces = batch['faces'].to(self.device).squeeze(0)
            masks = batch['masks'].to(self.device).squeeze(0)
            labels = batch['labels'].to(self.device)

            # # print shapes
            # print('voices: ', voices.shape)
            # print('faces: ', faces.shape)
            # print('masks: ', masks.shape)
            # print('labels: ', labels.shape)

            # check for nan and inf in faces, or voices.
            if torch.isnan(faces).any() or torch.isinf(faces).any():
                print('faces has nan or inf')
                print(asshole)
            if torch.isnan(voices).any() or torch.isinf(voices).any():
                print('voices has nan or inf')
                print(voices)
                print(asshole)

        
            # Forward pass
            with torch.no_grad():  # Encoder remains frozen
                voice_embs = self.encoder.voice_encoder(voices)
                face_embs = self.encoder.face_encoder(faces)    
                # # Detect placeholder frames (originally -1) and zero their embeddings
                # is_placeholder = (faces == -1).any(dim=-1, keepdim=True)  # (B, S, 1)
                # face_embs = torch.where(is_placeholder, torch.zeros_like(face_embs), face_embs)
            logits = self.decoder(face_embs, voice_embs, mask=masks)
            
            # Compute loss
            loss, pred_score, _, _ = self.loss_fn(
                logits,
                labels
            )
            # print('***************************************88')
            # print('')
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # Update tracking
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Free memory
            del voices, faces, labels, logits
            torch.cuda.empty_cache()
            
        return total_loss / len(self.train_loader)
    
    @staticmethod
    def _eval_operations(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        """Calculate classification metrics from model scores and ground truth labels.
        
        Args:
            scores: 2D array of batchwise model confidence scores for positive class
            labels: 2D array of batchwise binary ground truth labels (0/1)
            
        Returns:
            tuple: (optimal F1 score, balanced average precision, max score accuracy)
            
        Raises:
            ValueError: If inputs are invalid
            
        Note:
            Returns (0.0, 0.0, 0.0) when metrics cannot be computed due to:
            - Single-class labels
            - Empty inputs
        """
        # Find max score accuracy
        max_score_acc = 0
        for batch_scores, batch_labels in zip(scores, labels):
            max_index = np.argmax(batch_scores)
            correct_index = np.where(batch_labels == 1)[0]
            if max_index == correct_index:
                max_score_acc += 1
        
        max_score_acc /= len(scores)
        # unravel scores and labels and make arrays
        scores = np.concatenate(scores)
        labels = np.concatenate(labels)

        # Input validation
        if len(scores) != len(labels):
            raise ValueError("scores and labels must have same length")
        if len(scores) == 0:
            return (0.0, 0.0)
            
        # Handle single-class scenario
        if len(np.unique(labels)) < 2:
            return (0.0, 0.0)
            
        try:
            bap = average_precision_score(labels, scores)
            precision, recall, thresholds = precision_recall_curve(labels, scores)
            
            # Handle empty thresholds case
            if len(thresholds) == 0:
                return (0.0, bap)
                
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
            optimal_f1 = f1_scores.max() if len(f1_scores) > 0 else 0.0
            
        except Exception as e:
            print(f"Metric calculation failed: {str(e)}")
            return (0.0, 0.0)

        return (optimal_f1, bap, max_score_acc)

    
    def validate(self) -> Tuple[float, float, float]:
        """Run validation and return loss with optimal F1 score and balanced AP.
        
        Returns:
            tuple: (average_loss, average precision, max score accuracy)
        """
        self.decoder.eval()
        all_scores = []
        all_labels = []
        total_loss = 0
        n_speakers = []
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Data preparation remains unchanged
                voices = batch['voices'].to(self.device)
                faces = batch['faces'].to(self.device).squeeze(0)
                masks = batch['masks'].to(self.device).squeeze(0)
                labels = batch['labels'].to(self.device)
                # print(labels)
                
                # Forward pass
                voice_embs = self.encoder.voice_encoder(voices)
                face_embs = self.encoder.face_encoder(faces)
                # # Detect placeholder frames (originally -1) and zero their embeddings
                # is_placeholder = (faces == -1).any(dim=-1, keepdim=True)  # (B, S, 1)
                # face_embs = torch.where(is_placeholder, torch.zeros_like(face_embs), face_embs)

                logits = self.decoder(face_embs, voice_embs, mask=masks)
                # print(logits.shape)
                # Compute loss
                loss, pred_score, _, _ = self.loss_fn(logits, labels)
                total_loss += loss.item()
                # print(pred_score.shape)
                # print(pred_score)
                # print(asshle)
                # Collect scores and labels
                scores = pred_score[0,:].cpu().numpy().flatten()
                # print(scores)
                # print('')
                all_scores.append(scores)
                all_labels.append(labels.cpu().numpy().flatten())

                n_speakers.append(len(scores))
        # Convert to numpy arrays for metrics calculation
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate metrics
        _, bap, max_score_acc = self._eval_operations(all_scores, all_labels)
        return avg_loss, bap, max_score_acc
    
    def run_inference(self, utt_id2concurrent_tracks: Dict[str, Dict[str, Any]]):
        """Run inference on test set and save predictions with ASD output integration."""
        self.decoder.eval()
        save_path = self.config['output_path']
        os.makedirs(save_path, exist_ok=True)
        
        trackids2scores = {}
        max_score_acc = []

        # 1. Generate predictions for all utterances
        for batch in tqdm(self.test_loader, desc="Inference"):
            voices = batch['voices'].to(self.device)
            faces = batch['faces'].to(self.device).squeeze(0)
            masks = batch['masks'].to(self.device).squeeze(0)
            utt_id = batch['utt_ids'][0]
            spk_ids = batch['spk_ids']

            # Forward pass
            with torch.no_grad():
                voice_embs = self.encoder.voice_encoder(voices)
                face_embs = self.encoder.face_encoder(faces)
                logits = self.decoder(face_embs, voice_embs, mask=masks)

            # Get probability scores
            scores = self.loss_fn(logits).squeeze(0)

            # Check for nan/inf
            if np.isnan(scores).any() or np.isinf(scores).any():
                print(f'Skipping {utt_id} due to invalid scores')
                continue

            # Calculate accuracy
            gt_pid = utt_id.split('id-')[-1]
            max_index = np.argmax(scores)
            max_score_acc.append(1 if int(spk_ids[max_index]) == int(gt_pid) else 0)

            pid2score = {int(pid): score for pid, score in zip(spk_ids, scores)}
            concur_trackids = list(utt_id2concurrent_tracks[utt_id].keys())

            # Calculate concurrent track count per frame
            from collections import defaultdict
            frame_counter = defaultdict(int)
            for trackid in concur_trackids:
                for frame in utt_id2concurrent_tracks[utt_id][trackid]:
                    if frame['concurrent']:
                        frame_counter[frame['frame']] += 1

            # Process all concurrent tracks
            for trackid in concur_trackids:
                if trackid not in trackids2scores:
                    trackids2scores[trackid] = []
                    
                pid = utt_id2concurrent_tracks[utt_id][trackid][0]['pid']
                track_data = utt_id2concurrent_tracks[utt_id][trackid]

                for frame in track_data:
                    current_frame = frame['frame']
                    num_concurrent = frame_counter.get(current_frame, 0)
                    
                    if frame['concurrent']:
                        entry = {
                            'frame': current_frame,
                            'score': float(pid2score[int(pid)]),
                            'activity': int(frame['activity']),
                            'pid': int(pid),
                            'num_concurrent': num_concurrent  # New key
                        }
                    else:
                        entry = {
                            'frame': current_frame,
                            'score': 0.0,
                            'activity': int(frame['activity']),
                            'pid': int(pid),
                            'num_concurrent': num_concurrent  # New key
                        }
                    
                    trackids2scores[trackid].append(entry)

        # Post-process to remove duplicate frames
        for trackid, frame_entries in trackids2scores.items():
            unique_frames = {}
            for entry in frame_entries:
                frame = entry['frame']
                if frame in unique_frames:
                    if entry['score'] > unique_frames[frame]['score']:
                        unique_frames[frame] = entry
                else:
                    unique_frames[frame] = entry
            trackids2scores[trackid] = sorted(unique_frames.values(), key=lambda x: x['frame'])

        # Validate track lengths
        assert all(len(frames) <= 305 for frames in trackids2scores.values()), "Track length exceeds 305 frames"

        # Save results
        for trackid, scores in trackids2scores.items():
            dest_file = os.path.join(save_path, f"{trackid}.json")
            os.makedirs(os.path.dirname(dest_file), exist_ok=True)
            with open(dest_file, 'w') as f:
                json.dump(scores, f, indent=4)

        print(f'Max score accuracy: {sum(max_score_acc)/len(max_score_acc):.4f}')
        print(f"Inference completed. Results saved to {save_path}")
        
    def load_checkpoint(self, checkpoint_path: str):
        """
        Loads model checkpoint including decoder parameters, optimizer, scheduler, and training state.
        
        Args:
            checkpoint_path: Path to the checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        # print keys of checkpoint
        print(checkpoint.keys())

        # Load decoder parameters
        self.decoder.load_state_dict(checkpoint)
        
        # Load optimizer and scheduler states if available
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state if applicable
        if hasattr(self, 'epoch') and 'epoch' in checkpoint:
            self.epoch = checkpoint['epoch']
        if hasattr(self, 'best_bap') and 'best_bap' in checkpoint:
            self.best_bap = checkpoint['best_bap']
        
        print(f"Checkpoint loaded successfully from {checkpoint_path}")

    def run(self):
        """Main training loop"""
        for epoch in range(self.config['epochs']):
            self.epoch = epoch
            
            # Training phase
            train_loss = self.train_epoch()
            self.scheduler.step()
            
            # Validation phase
            val_loss, val_bap, val_MSA = self.validate()
            
            # Log metrics
            if self.config['wandb']:
                wandb.log({
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/bAP": val_bap * 100,
                    "epoch": epoch,
                    "lr": self.scheduler.get_last_lr()[0]
                })
            
            # Print epoch summary
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}", 
                  f"Val Loss: {val_loss:.4f}",
                    f"Val bAP: {val_bap * 100:.2f}", 
                    f"Val MSA: {val_MSA * 100:.2f}")

            # Save best model
            if val_bap > self.best_bap:
                self.best_bap = val_bap
                self.best_bap_epoch = epoch
                os.makedirs(self.config['save_dir'], exist_ok=True)
                torch.save(self.decoder.state_dict(), 
                         os.path.join(self.config['save_dir'], f"AP_[{round(val_bap,2)*100}%].pth"))
                
            # Early stopping
            if (epoch - self.best_bap_epoch) > self.config['patience']:
                print(f"Early stopping at epoch {epoch}")
                break
