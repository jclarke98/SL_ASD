from pathlib import Path

class Config:
    def __init__(self, **kwargs):
        paths = ['audio_direc', 'output_direc', 'bigAnnotPath', 'img_path', 'annot_path']
        for key, value in kwargs.items():
            if key in paths:
                value = Path(value)
            setattr(self, key, value)