"""Saved-distribution conversion must preserve hypotheses, masks and padding."""
import numpy as np
import pytest
import torch

from src.tasks.ball_refiner.refiner_3d.diffusion.data import rally_window
from src.tasks.ball_refiner.refiner_3d.diffusion.losses import (
    LossConfig,
    trajectory_loss,
)
from src.tasks.ball_refiner.refiner_3d.diffusion.model import DenoiserOutput
from src.utils.schema.court_normalization import normalize_court_position


def arrays():
    frames, modes, components = 9, 4, 125
    return {
        'gmm3d_means_m':np.ones((frames,components,3),np.float32),
        'gmm3d_covariance_m2':np.tile(np.eye(3,dtype=np.float32),(frames,components,1,1)),
        'gmm3d_weights':np.full((frames,components),1/components,np.float32),
        'gmm3d_camera_subsets':np.ones((frames,components,3),bool),
        'prior_only_probability':np.zeros(frames,np.float32),
        'timestamps_seconds':np.arange(frames)*1001/60000,
        'positions_3d_m':np.tile([0.,0.,2.],(frames,1)).astype(np.float32),
        'event_labels':np.zeros((frames,2),bool),
        'free_flight_mask':np.array([False]*6+[True]*3),
        'integration_converged':np.array([False]+[True]*8),
        'gmm2d_means_uv':np.full((3,frames,modes,2),.5,np.float32),
        'gmm2d_scale_tril_uv':np.tile(np.array([[.01,0],[.005,.02]],np.float32),(3,frames,modes,1,1)),
        'gmm2d_mixture_logits':np.broadcast_to(np.log([.1,.2,.3,.4]),(3,frames,modes)).astype(np.float32).copy(),
        'gmm2d_presence_logits':np.zeros((3,frames),np.float32),
        'source_size_wh':np.array([[101,51]]*3),
        'camera_estimated_K':np.tile(np.diag([100.,100.,1.]),(3,1,1)),
        'camera_estimated_R':np.tile(np.eye(3),(3,1,1)),
        'camera_estimated_t':np.zeros((3,3)),
    }


def test_all_125_components_and_pixel_covariance_survive_padding_and_gaps():
    data=arrays()
    record={'frames':9,'physics':{'gravity':9.81}}
    batch=rally_window(data,record,start=0,frames=12,allow_nonconverged=True)
    assert batch.condition.means_m.shape==(1,12,125,3)
    assert batch.means_2d_px.shape==(1,3,12,4,2)
    torch.testing.assert_close(batch.means_2d_px[0,0,0,0],torch.tensor([50.,25.]))
    torch.testing.assert_close(batch.covariance_2d_px2[0,0,0,0],torch.tensor([[1.,.25],[.25,1.0625]]))
    torch.testing.assert_close(batch.weights_2d[0,0,0],torch.tensor([.1,.2,.3,.4]))
    assert batch.condition.padding_mask[0].tolist()==[False]*9+[True]*3
    assert batch.free_flight_mask[0].tolist()==[False]*6+[True]*3+[False]*3
    # Nonconverged frame zero remains an actual attention/loss frame.
    assert not batch.condition.padding_mask[0,0]
    output=DenoiserOutput(normalize_court_position(batch.target_positions_m),torch.zeros((1,12,2)))
    _,terms=trajectory_loss(output,batch,LossConfig(1,.01,.0001,.1))
    assert terms['x0'].item()==0
    assert all(torch.isfinite(term) for term in terms.values())


def test_missing_or_nonconverged_flags_require_explicit_action():
    data=arrays()
    record={'frames':9,'physics':{'gravity':9.81}}
    with pytest.raises(ValueError,match='Nonconverged'):
        rally_window(data,record,start=0,frames=9,allow_nonconverged=False)
    del data['integration_converged']
    with pytest.raises(ValueError,match='Reintegrate'):
        rally_window(data,record,start=0,frames=9,allow_nonconverged=True)
