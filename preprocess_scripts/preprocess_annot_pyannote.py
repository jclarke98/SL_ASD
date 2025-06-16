#!/usr/bin/env python3
"""
preprocess_annot_pyannote.py

This script implements a complete segmentation pipeline using pyannote speaker diarisation 
with groundtruth (clipwise) identity annotation labels. Can also process the images if enabled.

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
import json
import yaml
import torch
from pathlib import Path
from tqdm import tqdm
import shutil

# Utility imports
from preprocess_utils import (
    Config,
    AudioSegmentClipper,
    SpeakerMatcher,
    format_gt,
    load_annotations,
    process_image_data,
    create_uttid2clipid,
    create_audio_manifest,
    create_data_splits
)

from pyannote.audio import Pipeline

from time import time


def is_speaker_visible(video, seg, fps=30):
    """Determine if speaker was visible during their utterance.
    
    Args:
        video: The video annotation dictionary
        seg: Current social segment dictionary
        default_fps: Fallback FPS if not found in video data
    
    Returns:
        bool: True if speaker was visible during utterance, False otherwise
    """
    person_id = seg["person"]
    seg_start = seg["start_time"]
    seg_end = seg["end_time"]
    
    # Find matching person in video annotations
    for person in video.get('persons', []):
        if person.get('person_id') == person_id:
            # Check all tracking paths for this person
            for track in person.get('tracking_paths', []):
                for entry in track.get('track', []):
                    # Convert video frame to time if available
                    if 'frame' in entry and entry['frame'] is not None:
                        entry_time = entry['frame'] / fps
                        if seg_start <= entry_time <= seg_end:
                            return True
    return False

# =============================================================================
# Core Processing Function
# =============================================================================
def process_audio_file(video, config, pipeline):
    """
    Process a single 5-minute audio file:
      1) Run pyannote diarization on the raw audio.
      2) Match the hypothesized segments to ground truth utterance boundaries.
      3) Clip and save the utterances.
      
    Args:
        video (dict): Video annotation dict containing a unique 'video_uid', path info, and ground truth segments.
        config (Config): Configuration parameters.
        pipeline (pyannote.audio.Pipeline): Pre-loaded pyannote diarization pipeline.
    
    Returns:
        correspondence (dict): Mapping of utterance IDs to clip information.
    """
    # Assume video['video_uid'] matches the audio file name (e.g. "video123.wav")
    video_id = video["clip_uid"]
    audio_path = config.audio_direc / f"{video_id}.wav"
    if not audio_path.exists():
        print(f"Audio file {audio_path} not found. Skipping.")
        return {}, None

    # Run pyannote diarization on the entire audio file.
    # The pipeline returns a pyannote.core.Annotation object.
    diarization = pipeline(str(audio_path))
    
    # Convert diarization output to a list of segment dicts.
    # Each segment has a start, end, and an assigned speaker label.
    hyp_segments = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        duration = segment.end - segment.start
        if duration >= config.min_utterance_duration:
            hyp_segments.append({
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker
            })
    
    # Load ground truth utterance boundaries for this video.
    # Assume video contains a key "social_segments_talking" with list of segments.
    gt_speech = {}
    for seg in video.get("social_segments_talking", []):
        # check if the utterance occurs on-screen
        if not is_speaker_visible(video, seg):
            continue
        person = seg["person"]
        # if person in {"0", "-1"}: # add way of skipping utterances which appear off-screen
        #     continue
        start = seg["start_time"]
        end = seg["end_time"]
        gt_speech.setdefault(person, []).append((start, end))
    gt_formatted = format_gt(gt_speech)

    # Use SpeakerMatcher to align hypothesized segments with ground truth.
    matcher = SpeakerMatcher(hyp_segments, gt_formatted)
    matcher.compute_overlap_matrix()
    matcher.find_optimal_assignment(config.overlap_threshold)
    matcher.assign_speaker_ids()
    labeled_segments = matcher.get_labeled_segments()

    # Filter out segments that were not matched (i.e. have no assigned speaker)
    final_segments = [seg for seg in labeled_segments if seg["speaker_id"] is not None]
    video_id = video["clip_uid"]
    audio_path = config.audio_direc / f"{video_id}.wav"

    # Define custom ID formatter for pyannote-style segments
    def pyannote_formatter(basename, start, end, speaker_id):
        start_str = f"{start:.1f}".replace('.', '_')
        end_str = f"{end:.1f}".replace('.', '_')
        return f"{basename}_clipped-{start_str}:{end_str}_id-{speaker_id}"

    clip_output_dir = config.output_direc / "audio" / video_id
    try:
        clipper = AudioSegmentClipper(audio_path, final_segments, clip_output_dir, pyannote_formatter)
        correspondence = clipper.process_all(min_duration=config.min_utterance_duration)
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        correspondence = {}
    return correspondence, video_id

def main():
    # Load config
    BASE_DIR = Path(__file__).resolve().parent.parent
    CONFIG_PATH = BASE_DIR / "configs" / "preprocess_annot.yml"
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)['pyannote']
    config = Config(**config)
    config.output_direc = str(config.output_direc) + '_' + str(config.overlap_threshold)
    os.makedirs(config.output_direc, exist_ok=True)
    
    # empty the audio directory
    audio_dir = config.output_direc / "audio"
    if audio_dir.exists():
        print('Deleting and recreating audio directory...')
        shutil.rmtree(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth annotations.
    annotations, clip2split = load_annotations(config.bigAnnotPath)
    
    # Load the pyannote diarization pipeline.
    print("Loading pyannote diarization pipeline...")
    pipeline = Pipeline.from_pretrained(
        config.pyannote_pipeline, 
        use_auth_token=config.hf_auth_token
                                        )
    pipeline.to(torch.device(config.pyannote_device))


    uttid2spkid_and_fpath = {}
    uttid2parent_vid = {}
    
    # Process each video (audio file) in the annotations.
    for video in tqdm(annotations, desc="Processing Audio Files"):
        for clip in video['clips']:
            correspondence, video_id = process_audio_file(clip, config, pipeline)
            if correspondence:
                uttid2spkid_and_fpath.update(correspondence)
                for uttid in correspondence:
                    # Record which video and split (train/val) this utterance came from.
                    uttid2parent_vid[uttid] = clip2split.get(clip['clip_uid'], "train")

    # Create manifest files.
    annot_dir = config.output_direc / "annot"
    os.makedirs(annot_dir, exist_ok=True)
    create_uttid2clipid(uttid2spkid_and_fpath, annot_dir)
    create_audio_manifest(uttid2spkid_and_fpath, annot_dir)
    create_data_splits(uttid2parent_vid, annot_dir, config.val_ratio)
    
    print('Audio processing complete.')

    # Process images if enabled.
    if config.process_images:
        process_image_data(config)

        print("Processing complete.")

if __name__ == "__main__":
    main()