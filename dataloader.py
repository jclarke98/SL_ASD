import json
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class SpeakerAwareDataset(Dataset): 
    """
    Dataset class to load batches containing utterances from the 
    same clip and speaker, along with face embeddings from all speakers in the clip.
    """

    def __init__(self, utt_ids, name2voice_emb, name2face_emb, uttid2clipid, 
             clip_id2spk_id2img_paths, batch_size, split, visible_only=False,
             utt_id2concurrent_tracks=None):
        """
        Args:
            utt_ids (list): List of utterance ids.
            name2voice_emb (dict): Maps utterance file paths to voice embeddings.
            name2face_emb (dict): Maps image file paths to face embeddings.
            uttid2clipid (dict): Maps utterance IDs to clip IDs.
            clip_id2spk_id2img_paths (dict): Maps clip IDs to speaker IDs and their image paths.
            batch_size (int): Total number of face embeddings per batch.
            split (str): One of 'train', 'val', 'test', or 'inference'.
            visible_only (bool): Whether to use only visible speakers.
            utt_id2concurrent_tracks (dict, optional): Mapping from utterance ids to concurrent track annotations.
                If provided and split=='inference', only utterances whose target speaker has a concurrent face track
                are included.
        """
        self.name2voice_emb = name2voice_emb
        self.name2face_emb = name2face_emb
        self.uttid2clipid = uttid2clipid
        self.clip_id2spk_id2img_paths = clip_id2spk_id2img_paths
        self.batch_size = batch_size
        self.utt_ids = utt_ids
        self.split = split
        self.visible_only = visible_only
        self.utt_id2concurrent_tracks = utt_id2concurrent_tracks

        # Precompute batches based on split type
        if split == 'train':
            batches = self._create_batches()
        elif split == 'val' or split == 'test':
            batches = self._create_single_utterance_batches()
        elif split == 'inference' and utt_id2concurrent_tracks is not None:
            batches = self._create_inference_batches()
        else:
            batches = self._create_single_utterance_batches()

        self.batches = batches
        print('average number of faces per batch: ',
            np.mean([len(batch['image_info']) for batch in self.batches]),
            'in:', split)
        ids_per_utt = []
        for batch in self.batches:
            n_ids = set()
            for spk_id, _ in batch['image_info']:
                n_ids.update(spk_id)
            ids_per_utt.append(len(n_ids))
        print('utterances stratified by number of distinct faces per batch: ')
        print(pd.value_counts(ids_per_utt))

    def _parse_speaker_id(self, utt_id):
        """Extracts speaker ID from utterance ID (assumes '_id-' suffix)."""
        parts = utt_id.split('_id-')
        if len(parts) < 2:
            raise ValueError(f"Unable to parse speaker ID from utterance ID: {utt_id}")
        speaker_part = parts[-1].split('.')[0]  # Remove file extension if present
        return speaker_part

    def _create_single_utterance_batches(self):
        batches = []

        for utt_id in self.utt_ids:
            clip_id = self.uttid2clipid.get(utt_id)
            if not clip_id:
                continue  # Skip if clip ID not found

            try:
                speaker_id = self._parse_speaker_id(utt_id)
            except ValueError:
                continue

            if self.visible_only:
                if speaker_id in {"0", "-1"}:  # Skip utterances from camera wearer and never-visible voices
                    # print('non-visible speaker detected and skipping')
                    continue
            else:
                if speaker_id in {"-1"}:
                    continue

            # Get all speakers in the clip
            spk_id2img_paths = self.clip_id2spk_id2img_paths.get(clip_id, {})
            num_speakers = len(spk_id2img_paths)
            if num_speakers <= 1:
                continue  # Need at least 2 speakers for meaningful comparison

            # Calculate image distribution per speaker
            images_per_speaker = self.batch_size // num_speakers
            remainder = self.batch_size % num_speakers

            # Sample images for each speaker
            image_groups = []
            speaker_ids = list(spk_id2img_paths.keys())

            for i, spk_id in enumerate(speaker_ids):
                sample_size = images_per_speaker + (1 if i < remainder else 0)
                available_imgs = spk_id2img_paths.get(spk_id, [])

                if not available_imgs:
                    continue  # Skip speakers without images

                if len(available_imgs) < sample_size:
                    sampled = available_imgs  # Use all available if insufficient
                else:
                    sampled = random.sample(available_imgs, sample_size)

                image_groups.append((spk_id, sampled))

            if not image_groups:
                continue

            batches.append({
                'utt_ids': [utt_id],  # Single utterance per batch
                'speaker_id': speaker_id,
                'image_info': image_groups
            })

        return batches

    def _create_batches(self):
        batches = []

        # Group utterances by (clip_id, speaker_id)
        clip_speaker_groups = defaultdict(list)
        for utt_id in self.utt_ids:
            clip_id = self.uttid2clipid.get(utt_id)
            if not clip_id:
                continue  # Skip if clip ID not found
            try:
                speaker_id = self._parse_speaker_id(utt_id)
            except ValueError:
                continue
            clip_speaker_groups[(clip_id, speaker_id)].append(utt_id)

        # Create batches for each (clip_id, speaker_id) group
        for (clip_id, speaker_id), utt_group in clip_speaker_groups.items():
            if self.visible_only:  # Skip utterances from camera wearer and never-visible voices
                if speaker_id in {"0", "-1"}:
                    continue
            else:
                if speaker_id in {"-1"}:
                    continue

            # Get all speakers in the clip
            spk_id2img_paths = self.clip_id2spk_id2img_paths.get(clip_id, {})
            if not spk_id2img_paths:
                continue  # Skip if no speaker data for clip

            num_speakers = len(spk_id2img_paths)
            images_per_speaker = self.batch_size // num_speakers
            remainder = self.batch_size % num_speakers

            image_groups = []
            speaker_ids = list(spk_id2img_paths.keys())
            for i, spk_id in enumerate(speaker_ids):
                sample_size = images_per_speaker + (1 if i < remainder else 0)
                available_imgs = spk_id2img_paths.get(spk_id, [])
                if not available_imgs:
                    continue  # Skip speaker with no images

                if len(available_imgs) < sample_size:
                    sampled = available_imgs
                else:
                    sampled = random.sample(available_imgs, sample_size)
                image_groups.append((spk_id, sampled))

            batches.append({
                'utt_ids': utt_group,
                'speaker_id': speaker_id,
                'image_info': image_groups
            })

        return batches

    def _create_inference_batches(self):
        print('creating inference batches')
        """Creates batches only for utterances whose target speaker has at least one concurrent face track."""
        batches = []

        for utt_id in self.utt_ids:
            clip_id = self.uttid2clipid.get(utt_id)
            if not clip_id:
                continue

            try:
                speaker_id = self._parse_speaker_id(utt_id)
            except ValueError:
                continue

            # Check if there is a concurrent track for the target speaker
            concurrent_info = self.utt_id2concurrent_tracks.get(utt_id, {})
            target_has_concurrent = False
            for track_id, annotations in concurrent_info.items():
                # Assuming the track id contains the speaker id in the format ":{speaker_id}:"
                if f":{speaker_id}:" in track_id:
                    if any(ann.get("concurrent", False) for ann in annotations):
                        target_has_concurrent = True
                        break
            if not target_has_concurrent:
                continue  # Skip this utterance if no concurrent face track for the target speaker

            if self.visible_only:
                if speaker_id in {"0", "-1"}:
                    # print('non-visible speaker detected and skipping')
                    continue
            else:
                if speaker_id in {"-1"}:
                    continue

            spk_id2img_paths = self.clip_id2spk_id2img_paths.get(clip_id, {})
            num_speakers = len(spk_id2img_paths)
            # if num_speakers <= 1:
            #     continue

            images_per_speaker = self.batch_size // num_speakers
            remainder = self.batch_size % num_speakers

            image_groups = []
            speaker_ids = list(spk_id2img_paths.keys())
            for i, spk_id in enumerate(speaker_ids):
                sample_size = images_per_speaker + (1 if i < remainder else 0)
                available_imgs = spk_id2img_paths.get(spk_id, [])
                if not available_imgs:
                    continue

                if len(available_imgs) < sample_size:
                    sampled = available_imgs
                else:
                    sampled = random.sample(available_imgs, sample_size)
                image_groups.append((spk_id, sampled))

            if not image_groups:
                continue

            batches.append({
                'utt_ids': [utt_id],
                'speaker_id': speaker_id,
                'image_info': image_groups
            })

        return batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        batch = self.batches[idx]

        # Load voice embeddings
        voice_embs = [self.name2voice_emb.get(utt_id, np.full((192), 1)) for utt_id in batch['utt_ids']]
        voice_tensor = torch.stack([torch.from_numpy(emb) for emb in voice_embs]).to(torch.float32)

        # Process face embeddings for each speaker group
        max_samples = max([len(img_paths) for _, img_paths in batch['image_info']])
        if not self.visible_only:
            face_groups = [torch.full((1, 512), -1.0)]  # Placeholder for non-visible speakers
            labels = [1 if batch['speaker_id'] == "0" else 0]
            spk_ids = [0]
            masks = [[1] * max_samples]
        else:
            face_groups, labels, spk_ids, masks = [], [], [], []
        for spk_id, img_paths in batch['image_info']:
            face_embs = [self.name2face_emb[img_path] for img_path in img_paths]
            face_tensor = torch.stack([torch.from_numpy(emb) for emb in face_embs])
            valid_frames = len(img_paths)

            face_groups.append(face_tensor)
            spk_ids.append(int(spk_id))
            labels.append(1 if spk_id == batch['speaker_id'] else 0)
            mask = [0] * valid_frames + [1] * (max_samples - valid_frames)
            masks.append(mask)

        padded_faces = []
        for face_tensor in face_groups:
            pad_size = max_samples - face_tensor.size(0)
            if pad_size > 0:
                padded = torch.cat([face_tensor, torch.zeros(pad_size, face_tensor.size(1))], dim=0)
            else:
                padded = face_tensor
            padded_faces.append(padded)

        faces_tensor = torch.stack(padded_faces)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        masks_tensor = torch.tensor(masks, dtype=torch.bool)

        if self.split in ['train', 'val']:
            return {
                'voices': voice_tensor,
                'faces': faces_tensor,
                'labels': labels_tensor,
                'masks': masks_tensor
            }
        elif self.split in ['test', 'inference']:
            return {
                'voices': voice_tensor,
                'faces': faces_tensor,
                'masks': masks_tensor,
                'spk_ids': spk_ids,
                'utt_ids': batch['utt_ids'][0]
            }

def load_annotations(uttid2clipid_path, clip_id2spk_id2img_paths_path):
    """Load annotation files."""
    with open(uttid2clipid_path, 'r') as f:
        uttid2clipid = json.load(f)
    with open(clip_id2spk_id2img_paths_path, 'r') as f:
        clip_id2spk_id2img_paths = json.load(f)
    return uttid2clipid, clip_id2spk_id2img_paths


def get_dataloader(utt_ids, name2voice_emb, name2face_emb, uttid2clipid, 
                   clip_id2spk_id2img_paths, batch_size, shuffle=True, num_workers=0,
                   split='train', visible_only=False, utt_id2concurrent_tracks=None):
    """Creates a DataLoader for the given utterance ids."""
    dataset = SpeakerAwareDataset(
        utt_ids, name2voice_emb, name2face_emb, uttid2clipid,
        clip_id2spk_id2img_paths, batch_size, split, visible_only,
        utt_id2concurrent_tracks=utt_id2concurrent_tracks
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Each dataset item is a pre-formed batch
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader