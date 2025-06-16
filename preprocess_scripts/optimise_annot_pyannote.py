#!/usr/bin/env python3
"""
preprocess_eval_der.py

This script runs a pyannote diarization pipeline on all clips, collects hypothesized segments
and ground-truth utterance boundaries, and then evaluates SpeakerMatcher efficacy in terms of
DER for a range of overlap thresholds.

It is heavily inspired by preprocess_annot_pyannote.py but refactored to:
  - Run diarization once per clip and store segments.
  - Iterate over a hard-coded list of overlap thresholds.
  - For each threshold, run matching and compute DER across all clips.
  - Print a summary of DER vs. threshold so you can pick the best threshold.
  - Keep modular structure so SpeakerMatcher can be modified independently.
"""

import os
import json
import yaml
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import shutil
import logging

from preprocess_utils import (
    Config,
    load_annotations,
    format_gt,
    # Note: SpeakerMatcher is imported and used below
)
from preprocess_utils import SpeakerMatcher  # keep SpeakerMatcher modular

from pyannote.audio import Pipeline
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Annotation, Segment

def is_speaker_visible(video, seg, fps=30):
    """
    Determine if speaker was visible during their utterance.

    Args:
        video: The video annotation dictionary
        seg: Current social segment dictionary
        fps: Frames per second, default 30
    Returns:
        bool: True if speaker was visible during utterance, False otherwise
    """
    person_id = seg["person"]
    seg_start = seg["start_time"]
    seg_end = seg["end_time"]

    for person in video.get('persons', []):
        if person.get('person_id') == person_id:
            for track in person.get('tracking_paths', []):
                for entry in track.get('track', []):
                    if 'frame' in entry and entry['frame'] is not None:
                        entry_time = entry['frame'] / fps
                        if seg_start <= entry_time <= seg_end:
                            return True
    return False

def extract_segments_for_all_clips(annotations, config, pipeline):
    """
    For each clip in annotations:
      - Run diarization once to get hypothesized segments.
      - Collect ground-truth utterance boundaries (filtered by visibility).
    Returns:
      data: a list of dicts, each with:
         'clip_id': str,
         'hyp_segments': list of dicts {'start': float, 'end': float, 'speaker': str},
         'gt_segments': list of dicts {'start': float, 'end': float, 'speaker_id': str}
    """
    data = []
    annotations = annotations[:]
    for video in tqdm(annotations, desc="Collecting segments"):
        for clip in video.get('clips', []):
            video_id = clip["clip_uid"]
            audio_path = config.audio_direc / f"{video_id}.wav"
            if not audio_path.exists():
                logging.warning(f"Audio file {audio_path} not found. Skipping clip {video_id}.")
                continue

            # Run pyannote diarization on the entire audio file.
            diarization = pipeline(str(audio_path))
            # Convert diarization output to list of segments
            hyp_segments = []
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                duration = segment.end - segment.start
                if duration >= config.min_utterance_duration:
                    hyp_segments.append({
                        "start": segment.start,
                        "end": segment.end,
                        "speaker": speaker
                    })

            # Load ground truth utterance boundaries for this clip
            gt_speech = {}
            for seg in clip.get("social_segments_talking", []):
                if not is_speaker_visible(clip, seg):
                    continue
                person = seg["person"]
                start = seg["start_time"]
                end = seg["end_time"]
                # ensure duration >= min_utterance_duration?
                if (end - start) < config.min_utterance_duration:
                    continue
                gt_speech.setdefault(person, []).append((start, end))

            gt_formatted = format_gt(gt_speech)  # expects list of dicts with 'start','end','speaker_id'

            if not gt_formatted:
                logging.info(f"No GT segments (visible) for clip {video_id}. Skipping.")
                continue

            data.append({
                'clip_id': video_id,
                'hyp_segments': hyp_segments,
                'gt_segments': gt_formatted
            })

    return data

def build_annotation_from_segments(segments, label_key):
    """
    Build a pyannote.core.Annotation from a list of segment dicts.

    Args:
      segments: list of dicts, each with at least 'start', 'end', and a label_key specifying speaker ID
      label_key: e.g. 'speaker_id' or 'speaker'
    Returns:
      annotation: pyannote.core.Annotation
    """
    ann = Annotation()
    for seg in segments:
        speaker_label = seg.get(label_key)
        if speaker_label is None:
            continue
        start = seg['start']
        end = seg['end']
        # Only add if valid duration
        if end > start:
            ann[Segment(start, end)] = str(speaker_label)
    return ann

def evaluate_thresholds(data, thresholds, config):
    """
    For each overlap threshold in thresholds:
      - For each clip in data:
          * Run SpeakerMatcher matching with that threshold
          * Build hypothesis Annotation from matched segments
          * Build reference Annotation from GT segments
          * Compute DER for that single clip in isolation
      - Average the per-clip DERs (simple arithmetic mean over clips with valid DER)
    Returns:
      results: dict mapping threshold -> {
          'average_der': float or None,
          'per_clip': { clip_id: der_value_or_None, ... }
      }
    """
    results = {}

    for thr in thresholds:
        logging.info(f"Evaluating threshold = {thr:.2f}")
        per_clip_ders = {}  # clip_id -> DER (float) or None

        for item in tqdm(data, desc=f"Threshold {thr:.2f}"):
            clip_id = item.get('clip_id', '<unknown>')
            hyp_segs = item['hyp_segments']
            gt_segs = item['gt_segments']

            # Run matching for this clip
            matcher = SpeakerMatcher(hyp_segs, gt_segs)
            matcher.compute_overlap_matrix()
            matcher.find_optimal_assignment(overlap_threshold=thr)
            matcher.assign_speaker_ids()
            labeled = matcher.get_labeled_segments()

            # Build reference Annotation
            ref_ann = build_annotation_from_segments(gt_segs, label_key='speaker_id')
            # Build hypothesis Annotation (only segments with assigned 'speaker_id')
            hyp_ann = build_annotation_from_segments(labeled, label_key='speaker_id')

            # Compute DER for this clip in isolation
            try:
                clip_metric = DiarizationErrorRate()
                clip_metric(ref_ann, hyp_ann)
                m = clip_metric[:]  # e.g. {'confusion': ..., 'false alarm': ..., 'missed detection': ..., 'correct': ..., 'total': ...}
                fa = m.get('false alarm', 0.0)
                missed = m.get('missed detection', 0.0)
                conf = m.get('confusion', 0.0)
                total = m.get('total', 0.0)
                if total > 0:
                    der_clip = (fa + missed + conf) / total
                    per_clip_ders[clip_id] = der_clip
                else:
                    # No reference speech duration: cannot compute DER meaningfully
                    logging.warning(f"Clip {clip_id}: total reference duration is zero; skipping DER for this clip.")
                    per_clip_ders[clip_id] = None
            except Exception as e:
                logging.warning(f"Error computing DER for clip {clip_id} at threshold {thr}: {e}")
                per_clip_ders[clip_id] = None

        # After all clips: compute average of per-clip DERs
        valid_ders = [v for v in per_clip_ders.values() if v is not None]
        if valid_ders:
            average_der = sum(valid_ders) / len(valid_ders)
        else:
            average_der = None
            logging.warning(f"No valid per-clip DERs computed for threshold {thr:.2f}; average DER is None.")

        logging.info(f"Threshold {thr:.2f} -> average DER over {len(valid_ders)}/{len(per_clip_ders)} clips: "
                     f"{(f'{average_der:.3f}' if average_der is not None else 'N/A')}")

        # Store results: average and per-clip breakdown
        results[thr] = {
            'average_der': average_der,
            'per_clip': per_clip_ders
        }

    return results

def main():
    # Load config
    BASE_DIR = Path(__file__).resolve().parent.parent
    CONFIG_PATH = BASE_DIR / "configs" / "preprocess_annot.yml"
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        cfg_dict = yaml.safe_load(f)['pyannote']
    config = Config(**cfg_dict)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load ground truth annotations.
    annotations, clip2split = load_annotations(config.bigAnnotPath)

    # Load pyannote pipeline
    logging.info("Loading pyannote diarization pipeline...")
    pipeline = Pipeline.from_pretrained(
        config.pyannote_pipeline,
        use_auth_token=config.hf_auth_token
    )
    pipeline.to(torch.device(config.pyannote_device))

    # Collect hypothesized and ground-truth segments for all clips
    data = extract_segments_for_all_clips(annotations, config, pipeline)
    if not data:
        logging.error("No clips yielded both hyp and GT segments. Exiting.")
        return

    # Define the list of overlap thresholds to evaluate
    # You can adjust this list to the thresholds you want to test
    thresholds = np.linspace(0.01, 0.99, num=100).tolist()  # [0.01, 0.11, ..., 0.91]

    # Evaluate DER across thresholds
    results = evaluate_thresholds(data, thresholds, config)
    
    # Print summary table
    print("\nSummary of DER vs. overlap threshold:")
    print("{:<10} {:<8} {:<8} {:<8} {:<8}".format("Threshold", "DER", "FA(s)", "Miss(s)", "Conf(s)"))
    for thr in sorted(results.keys()):
        metrics = results[thr]
        der = metrics.get('average_der', None)
        print("{:<10.2f} {:<8.3f}".format(thr, der if der is not None else float('nan')))

    # # Optionally: save results to JSON
    # out_path = config.output_direc / "der_evaluation_results.json"
    # try:
    #     os.makedirs(config.output_direc, exist_ok=True)
    #     # Convert keys to strings for JSON
    #     json_results = {f"{thr:.2f}": results[thr] for thr in results}
    #     with open(out_path, 'w') as f:
    #         json.dump(json_results, f, indent=2)
    #     logging.info(f"Saved DER results to {out_path}")
    # except Exception as e:
    #     logging.warning(f"Could not save results to {out_path}: {e}")

if __name__ == "__main__":
    main()