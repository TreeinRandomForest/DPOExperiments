from datasets import load_dataset
from pathlib import Path

import json

def process_chartqa(out_loc: str):
    ds = load_dataset("HuggingFaceM4/ChartQA")

    out_loc = Path(out_loc)
    img_dir = out_loc / "images"

    for split in ds:
        img_dir = out_loc / "images" / split
        img_dir.mkdir(parents=True, exist_ok=True)
    
        with open(out_loc / f"{split}_data.json", 'w') as f:
            for idx, d in enumerate(ds[split]):

                assert len(d['label']) == 1

                fname = img_dir / f'img_{idx}.png'
                d['image'].save(fname)

                elem = {
                    'prompt': d['query'],
                    'chosen': d['label'][0],
                    'rejected': "TBD",
                    'image_path': str(fname),
                    'metadata': {'human_or_machine': d['human_or_machine']}
                }
                f.write(json.dumps(elem) + "\n")