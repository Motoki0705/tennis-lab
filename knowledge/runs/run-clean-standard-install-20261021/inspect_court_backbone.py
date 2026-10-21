"""Compare the embedded court backbone with its separately acquired checkpoint on CPU."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--court', required=True, type=Path)
parser.add_argument('--backbone', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
torch.set_num_threads(2)
checkpoint = torch.load(args.court, map_location='cpu', weights_only=False, mmap=True)
prefix = 'model.encoder.backbone.module.'
embedded = {key.removeprefix(prefix): value for key, value in checkpoint['state_dict'].items() if key.startswith(prefix)}
backbone = torch.load(args.backbone, map_location='cpu', weights_only=True)
if 'model' in backbone:
    backbone = backbone['model']
result = {'court_filename': args.court.name, 'backbone_filename': args.backbone.name,
          'embedded_prefix': prefix, 'embedded_tensors': len(embedded), 'standalone_tensors': len(backbone),
          'same_keys': embedded.keys() == backbone.keys(),
          'same_tensors': embedded.keys() == backbone.keys() and all(torch.equal(embedded[key], value) for key, value in backbone.items()),
          'parameters': sum(value.numel() for value in backbone.values())}
for label, path in [('court', args.court), ('backbone', args.backbone)]:
    with path.open('rb') as stream:
        result[label + '_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
args.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
