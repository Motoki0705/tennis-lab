"""One real-dev forward/backward, fixed first eight train prefixes, no updates."""
import json
import resource
import time
from pathlib import Path

import torch

from src.tasks.ball_refiner.refiner_3d.diffusion.data import collate_windows, rally_window
from src.tasks.ball_refiner.refiner_3d.diffusion.dev_config import load_config
from src.tasks.ball_refiner.refiner_3d.diffusion.flow import training_objective
from src.tasks.ball_refiner.refiner_3d.diffusion.model import TrajectoryDenoiser
from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import sha256
from src.tasks.ball_refiner.refiner_3d.synthetic.dataset import SyntheticDataset

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).with_suffix('.json')
DATA = Path('/home/kamimura/projects/tennis-lab/data/ball_refiner/i936-dev-h-r9-s936')
CONFIG = ROOT / 'src/tasks/ball_refiner/refiner_3d/training_dev.yaml'
torch.set_num_threads(1)
config = load_config(CONFIG)
torch.manual_seed(config.seed)
source = SyntheticDataset(DATA)
records = sorted((r for r in source.records if r['split'] == 'train'), key=lambda r:r['rally_id'])[:8]
started = time.perf_counter()
batch = collate_windows([rally_window(source.load(r), r, start=0, frames=config.frames, allow_nonconverged=True) for r in records])
model = TrajectoryDenoiser(config.model)
loss, terms = training_objective(model, batch, config.loss, torch.Generator().manual_seed(config.seed+1), objective='flow')
assert torch.isfinite(loss)
loss.backward()
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
result = {'kind':'CPU real-dev plumbing; no optimizer update or quality conclusion',
    'rallies':[r['rally_id'] for r in records], 'config_sha256':sha256(CONFIG),
    'source_manifest_sha256':sha256(DATA/'manifest.json'),
    'shape':list(batch.condition.means_m.shape), 'real_frames':int((~batch.condition.padding_mask).sum()),
    'parameters':sum(p.numel() for p in model.parameters()), 'loss':loss.item(),
    'terms':{k:v.item() for k,v in terms.items()}, 'gradient_norm':norm.item(),
    'all_gradients_finite':True, 'seconds':time.perf_counter()-started,
    'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
OUTPUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result))
