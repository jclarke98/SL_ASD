#!/usr/bin/env python3
"""
preprocess_annot_groundtruth.py

This script implements segmentation using direct ground truth boundaries. Can also process the images if enabled.

output_dir/
    audio/
        clipid/
            clipid_clipped-0_0:5_0_id-0.wav
            clipid_clipped-0_0:5_1_id-1.wav
            ...
    annot/
        uttid2clipid.json
        audio_manifest.json
        splits.json
        image_manifest.json (if enabled)

uttid2clipid.json:
    {   "clipid_clipped-0_0:5_0_id-0": "clipid",
        ...
    }

audio_manifest.csv:
    uttid,clip_path,speaker_id,fpath
    clipid_clipped-0_0:5_0_id-0,clipid_clipped-0_0:5_0_id-0.wav,0,clipid.wav
    ...

splits.json:
    {   "train": ["clipid_clipped-0_0:5_0_id-0", ...],
        "val": ["clipid_clipped-0_0:5_1_id-1", ...],
        "test": ["clipid_clipped-0_0:5_2_id-2", ...]
    }

    
clip_id2spk_id2img_paths.json:
    {   "clipid": {
            "0": ["path/to/image1.jpg", "path/to/image2.jpg", ...],
            "1": ["path/to/image3.jpg", "path/to/image4.jpg", ...],
            ...
        },
        ... 
    }
"""


import os
import yaml
from pathlib import Path
from tqdm import tqdm

# Utility imports
from preprocess_utils import (
    Config,
    AudioSegmentClipper,
    format_gt,
    load_annotations,
    process_image_data,
    create_uttid2clipid,
    create_audio_manifest,
    create_data_splits
)

# =============================================================================
# Core Processing Function
# =============================================================================
def process_audio_file(clip, config):
    """Directly process ground truth segments from annotations."""
    audio_path = config.audio_direc / f"{clip['clip_uid']}.wav"
    if not audio_path.exists():
        print(f"Audio file {audio_path} not found. Skipping.")
        return {}, None

    # Extract and format ground truth segments
    gt_speech = {}
    for seg in clip.get("social_segments_talking", []):
        person = seg["person"]
        start = seg["start_time"]
        end = seg["end_time"]
        gt_speech.setdefault(person, []).append((start, end))
    formatted_segments = format_gt(gt_speech)

    # Filter by duration and process
    valid_segments = [s for s in formatted_segments 
                     if (s['end'] - s['start']) >= config.min_utterance_duration]
    
    clip_output_dir = config.output_direc / "audio_groundtruth" / clip['clip_uid']

    # Define custom ID formatter for ground truth segments
    def gt_formatter(basename, start, end, speaker_id):
        start_str = f"{start:.5f}".replace('.', '_')
        end_str = f"{end:.5f}".replace('.', '_')
        return f"{basename}_clipped-{start_str}:{end_str}_id-{speaker_id}"

    clipper = AudioSegmentClipper(audio_path, valid_segments, clip_output_dir, gt_formatter)
    correspondence = clipper.process_all(min_duration=config.min_utterance_duration)
    return correspondence, clip['clip_uid']

# =============================================================================
# Main Pipeline (Updated with utility imports)
# =============================================================================
def main():
    # Load config
    BASE_DIR = Path(__file__).resolve().parent.parent
    CONFIG_PATH = BASE_DIR / "configs" / "preprocess_annot.yml"
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)['gt']
    config = Config(**config)

    os.makedirs(config.output_direc, exist_ok=True)
    
    annotations, clip2split = load_annotations(config.bigAnnotPath)
    uttid2spkid_and_fpath = {}
    uttid2parent_vid = {}

    for video in tqdm(annotations, desc="Processing Audio Files"):
        for clip in video['clips']:
            correspondence, clip_uid = process_audio_file(clip, config)
            if correspondence:
                uttid2spkid_and_fpath.update(correspondence)
                for utt_id in correspondence:
                    uttid2parent_vid[utt_id] = clip2split.get(clip_uid, ("", "train"))

    # Create output manifests
    annot_dir = config.output_direc / "annot"
    create_uttid2clipid(uttid2spkid_and_fpath, annot_dir)
    create_audio_manifest(uttid2spkid_and_fpath, annot_dir)
    create_data_splits(uttid2parent_vid, annot_dir, config.val_ratio)

    print("Ground truth audio processing complete.")

    # Process images if enabled.
    if config.process_images:
        process_image_data(config)

        print("Processing images complete.")

if __name__ == "__main__":
    main()