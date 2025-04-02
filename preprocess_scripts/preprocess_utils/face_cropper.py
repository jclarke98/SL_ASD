import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from joblib import Parallel, delayed
from collections import defaultdict

import torchvision

def load_schema(annot_path: Path, split: str = 'train') -> pd.DataFrame:
    """Load trackwise annotation schema from CSV."""
    col_names = ['trackid', 'duration', 'fps', 'labels', 'start_frame']
    file_path = annot_path / f'active_speaker_{split}.csv'
    return pd.read_csv(file_path, names=col_names, delimiter='\t')

resize_size = 128
transform_fn = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(resize_size, resize_size)),
            torchvision.transforms.ToTensor()
        ])

def process_track(
    row: pd.Series,
    bbox_path: Path,
    img_path: Path,
    save_path: Path
) -> Tuple[str, str, List[Path]]:
    """Process a single track with enhanced image validation."""
    trackid = row['trackid']
    video_id = trackid[:36]
    valid_paths = []

    try:
        with open(bbox_path / f"{trackid}.json", 'r') as f:
            bboxes = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {trackid}.json: {str(e)}")
        return video_id, "", []

    save_dir = save_path / 'imgs' / video_id / trackid
    save_dir.mkdir(parents=True, exist_ok=True)
    pid = str(bboxes[0].get('pid', 'unknown'))

    for i, bbox in enumerate(bboxes):
        frame_num = min(int(bbox['frame']) + 1, 9997)
        img_src = img_path / video_id / f"img_{frame_num:05d}.jpg"
        img_dest = save_dir / f"{i:03d}.jpg"
        
        # Phase 1: Validate existing output
        if img_dest.exists():
            try:
                img_PIL = Image.open(img_dest)
                if img_PIL.mode != "RGB":
                    img_PIL = img_PIL.convert("RGB")
                data = transform_fn(img_PIL)
                assert data.shape == (3, 128, 128), img_dest
                valid_paths.append(img_dest)
                continue
            except Exception as e:
                print(f"Removing corrupted {img_dest}: {str(e)}")
                img_dest.unlink()
        # Load source with PIL for validation
        src_img = Image.open(img_src)
        if src_img.mode != "RGB":
            src_img = src_img.convert("RGB")
        img_array = np.array(src_img)

        x, y, w, h = (int(bbox[k]) for k in ('x', 'y', 'width', 'height'))

        # Crop and save
        cropped = Image.fromarray(img_array[y:y+h, x:x+w])
        if cropped.size[0] < 5 or cropped.size[1] < 5:
            print(f"small crop: {img_src}, therefore skipping")
            continue
        cropped.save(
            img_dest,
            quality=95,
            subsampling=0,
            optimize=True
        )
        valid_paths.append(img_dest)
    return video_id, pid, valid_paths

def crop_and_save_images(
    df: pd.DataFrame,
    bbox_path: Path,
    img_path: Path,
    save_path: Path
) -> Dict[str, Dict[str, List[Path]]]:
    """Process all tracks and return nested dictionary of saved paths."""
    results = Parallel(n_jobs=-1, prefer="threads")(
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