"""Independent C1/C2-to-render audit and complete decoded-video/PTS validation."""
from pathlib import Path
import json
import numpy as np
from src.utils.checksum import dual_sha256
from src.tennis_scene.visualization.player_view.camera_bundle import load_head_cameras
from src.tennis_scene.visualization.player_view.video import verify_video

root=Path('outputs/tennis_scene/visualize/player_view/blender-acceptance')
package=root/'package-v2'
delivery=root/'videos'
source=Path('../player-view-scene-bundle/outputs/tennis_scene/visualize/player_view/acceptance-final/bundle')
camera_source=Path('../player-view-head-camera/outputs/tennis_scene/visualize/player_view/head-camera-acceptance/cameras')
info=json.loads((package/'manifest.json').read_text())
manifest=json.loads((delivery/'manifest.json').read_text())
source_frames=np.load(source/'source_frames.npy')
np.testing.assert_array_equal(source_frames,np.arange(1010))
mesh=np.load(source/'player_smpl_valid.npy')
ball_valid=np.load(source/'ball_3d_valid.npy')
ball_positions=np.load(source/'ball_positions.npy')
assert info['fps']==59.94006 and manifest['fps']==59.94006
assert manifest['player_ids']==[0,1] and not manifest['preview']
assert manifest['source_frames']==source_frames.tolist()
assert manifest['package_sha256']==dual_sha256(package/'manifest.json')
source_hashes={}
for kind,directory in (('scene_files',source),('camera_files',camera_source)):
    source_hashes[kind]={name:dual_sha256(directory/name)==digest for name,digest in info['inputs'][kind].items()}
    assert all(source_hashes[kind].values())
summary={'source_frames':[0,1010],'fps':59.94006,'player_ids':[0,1],
         'source_digest_matches':source_hashes,'views':{},'videos':{},'human_acceptance':'pending'}
for index,identity in enumerate((0,1)):
    saved=load_head_cameras(camera_source,player_id=identity)
    report_path=delivery/f'render_player_{identity}.json'
    assert dual_sha256(report_path)==manifest['render_reports'][str(identity)]
    report=json.loads(report_path.read_text())
    assert report['complete'] and report['player_id']==identity and len(report['frames'])==1010
    assert report['package_sha256']==manifest['package_sha256']
    assert report['engine']=='CYCLES' and report['gpu_devices']==[{'name':'NVIDIA L4','type':'OPTIX'}]
    assert not report['motion_blur'] and report['dither_intensity']==0
    assert set(report['material'])=={'OpaqueGrayPlayer','OpaqueGreenBall'}
    assert all(m['alpha']==1 and m['emission_strength']==0 for m in report['material'].values())
    np.testing.assert_array_equal([r['source_frame'] for r in report['frames']],source_frames)
    np.testing.assert_array_equal([r['output_index'] for r in report['frames']],source_frames)
    np.testing.assert_array_equal([r['cache_index'] for r in report['frames']],source_frames)
    np.testing.assert_array_equal([r['camera_valid'] for r in report['frames']],saved.series.camera_valid)
    np.testing.assert_array_equal([r['camera_reason'] for r in report['frames']],saved.series.camera_rejection_code)
    camera_error=ball_error=pc2_error=0.0
    for frame,row in enumerate(report['frames']):
        ok=bool(saved.series.camera_valid[frame])
        expected_hidden=[not (ok and mesh[p,frame] and p!=index) for p in (0,1)]
        assert row['player_hidden']==expected_hidden
        assert row['ball_hidden']==(not (ok and ball_valid[frame]))
        if ok:
            expected=saved.series.camera_to_court[frame].copy()
            expected[:3,:3]*=np.array([1,-1,-1])
            error=float(np.max(np.abs(np.asarray(row['camera_matrix_world'])-expected)))
            assert error<2e-5
            camera_error=max(camera_error,error)
        else:
            assert row['camera_matrix_world'] is None
        if ball_valid[frame]:
            error=float(np.max(np.abs(np.asarray(row['ball_position'])-ball_positions[frame])))
            assert error<1e-5
            ball_error=max(ball_error,error)
        else:
            assert row['ball_position'] is None
        assert len(row['pc2_max_error_m'])==2
        error=float(np.max(row['pc2_max_error_m']))
        assert np.isfinite(error) and error<=1e-5
        pc2_error=max(pc2_error,error)
    missing=np.flatnonzero(~saved.series.camera_valid).tolist()
    assert manifest['camera_missing_frames'][str(identity)]==missing
    summary['views'][str(identity)]={'audited_frames':1010,'camera_valid':int(saved.series.camera_valid.sum()),
        'camera_missing_source_frames':missing,'visible_opponent_frames':sum(not r['player_hidden'][1-index] for r in report['frames']),
        'visible_ball_frames':sum(not r['ball_hidden'] for r in report['frames']),
        'max_camera_matrix_error':camera_error,'max_ball_position_error_m':ball_error,'max_reported_pc2_error_m':pc2_error,
        'denoising':report.get('denoising'),
        'source_masks_match':True,'viewer_hidden_all_frames':True,'dither_intensity':report['dither_intensity']}
for name,width,count in [('player_0.mp4',1280,1010),('player_1.mp4',1280,1010),('comparison.mp4',2560,1010),('comparison_short.mp4',2560,240)]:
    result=verify_video(delivery/name,frames=count,fps=59.94006,width=width,height=720)
    if name!='comparison_short.mp4':
        assert result['sha256']==manifest['verification'][name]['sha256']
    summary['videos'][name]=result
short=json.loads((delivery/'short_comparison.json').read_text())
assert short['verification']['decoded_frames']==240
assert short['source_frames']==[0,240]
assert short['verification']['sha256']==summary['videos']['comparison_short.mp4']['sha256']
summary['short_comparison_source_interval']={'start':0,'stop':240}
(root/'full_validation.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
