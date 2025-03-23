import os
import soundfile as sf
from pathlib import Path

class AudioSegmentClipper:
    def __init__(self, audio_path, segments, output_dir, id_formatter):
        self.audio_path = audio_path
        self.segments = segments
        self.output_dir = output_dir
        self.formatter = id_formatter
        self.correspondence = {}

    def process_all(self, min_duration=0.5):
        os.makedirs(self.output_dir, exist_ok=True)
        audio, sr = sf.read(str(self.audio_path))
        basename = self.audio_path.stem
        
        for seg in self.segments:
            if (seg['end'] - seg['start']) < min_duration:
                continue
                
            start = seg['start']
            end = seg['end']
            speaker_id = seg['speaker_id']
            
            utt_id = self.formatter(basename, start, end, speaker_id)
            out_path = self.output_dir / f"{utt_id}.wav"
            
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            sf.write(str(out_path), audio[start_sample:end_sample], sr)
            
            self.correspondence[utt_id] = {
                'file_path': str(out_path),
                'speaker_id': speaker_id
            }
        return self.correspondence