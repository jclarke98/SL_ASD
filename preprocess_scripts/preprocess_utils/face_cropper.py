import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import cv2
import pandas as pd
from joblib import Parallel, delayed
from collections import defaultdict

def load_schema(annot_path: Path, split: str = 'train') -> pd.DataFrame:
    """Load trackwise annotation schema from CSV."""
    col_names = ['trackid', 'duration', 'fps', 'labels', 'start_frame']
    file_path = annot_path / f'active_speaker_{split}.csv'
    return pd.read_csv(file_path, names=col_names, delimiter='\t')

def process_track(
    row: pd.Series,
    bbox_path: Path,
    img_path: Path,
    save_path: Path
) -> Tuple[str, str, List[Path]]:
    """Process a single track and return metadata with saved image paths."""
    trackid = row['trackid']
    video_id = trackid[:36]
    
    # Load bounding boxes
    bbox_file = bbox_path / f"{trackid}.json"
    try:
        with open(bbox_file, 'r') as f:
            bboxes = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {bbox_file}: {str(e)}")
        return video_id, "", []

    # Create output directory
    save_dir = save_path / 'imgs' / video_id / trackid
    save_dir.mkdir(parents=True, exist_ok=True)

    # Process each bounding box
    valid_paths = []
    pid = str(bboxes[0].get('pid', 'unknown'))
    
    for i, bbox in enumerate(bboxes):
        frame_num = min(int(bbox['frame']) + 1, 9997)  # Clamp frame number
        img_src = img_path / video_id / f"img_{frame_num:05d}.jpg"
        
        if not img_src.exists():
            continue

        # Generate output path
        img_dest = save_dir / f"{i:03d}.jpg"
        if img_dest.exists():
            valid_paths.append(img_dest)
            continue

        # Process image
        try:
            img = cv2.imread(str(img_src))
            x, y, w, h = (int(bbox[k]) for k in ('x', 'y', 'width', 'height'))
            cropped = img[y:y+h, x:x+w]
            
            if cropped.size == 0:
                print(f"Empty crop: {img_src}")
                continue
                
            cv2.imwrite(str(img_dest), cropped)
            valid_paths.append(img_dest)
        except Exception as e:
            print(f"Error processing {img_src}: {str(e)}")

    return video_id, pid, valid_paths

def crop_and_save_images(
    df: pd.DataFrame,
    bbox_path: Path,
    img_path: Path,
    save_path: Path
) -> Dict[str, Dict[str, List[Path]]]:
    """Process all tracks and return nested dictionary of saved paths."""
    results = Parallel(n_jobs=-1)(
        delayed(process_track)(row, bbox_path, img_path, save_path)
        for _, row in tqdm(df.iterrows(), total=len(df))
    )

    # Build nested dictionary structure
    output_dict = defaultdict(lambda: defaultdict(list))
    for clip_id, pid, paths in results:
        # convert paths to strings
        paths = [str(p) for p in paths]
        if pid and paths:
            output_dict[clip_id][pid].extend(paths)

    return output_dict