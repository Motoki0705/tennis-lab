"""CPU comparison of the official Market checkpoint and the previously used file."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.tasks.player_association.appearance.encoders import ClipReIDEncoder

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--official', required=True, type=Path)
parser.add_argument('--current', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
torch.set_num_threads(2)
crops = torch.rand((4, 3, 256, 128), generator=torch.Generator().manual_seed(42))
record = {'official_source': 'https://drive.google.com/file/d/1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7/view',
          'current_source': 'occurra/person_vit_clip_reid', 'device': 'cpu',
          'input_shape': list(crops.shape), 'input_kind': 'synthetic RGB in [0, 1]', 'seed': 42, 'files': []}
embeddings = []
for label, path in [('official', args.official), ('current', args.current)]:
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    encoder = ClipReIDEncoder('clipreid_vitb16_market1501', path, 'cpu')
    state = torch.load(path, map_location='cpu', weights_only=True)
    embedding = encoder.embed(crops)
    embeddings.append(embedding)
    record['files'].append({'label': label, 'filename': path.name, 'bytes': path.stat().st_size,
        'sha256': digest, 'image_encoder_keys': sum(key.startswith('image_encoder.') for key in state),
        'position_shape': list(state['image_encoder.positional_embedding'].shape), 'strict_visual_load': True,
        'embedding_shape': list(embedding.shape),
        'embedding_sha256': hashlib.sha256(embedding.numpy().tobytes()).hexdigest()})
    del encoder, state
record['embedding_equal'] = torch.equal(*embeddings)
record['embedding_max_abs_diff'] = (embeddings[0] - embeddings[1]).abs().max().item()
record['file_sha256_equal'] = record['files'][0]['sha256'] == record['files'][1]['sha256']
args.output.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
