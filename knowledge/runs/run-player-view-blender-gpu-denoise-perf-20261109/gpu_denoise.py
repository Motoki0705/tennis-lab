"""One-time C3 GPU OpenImageDenoise verification and fresh-first full delivery."""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import uuid
from datetime import datetime, timezone

import numpy as np
from PIL import Image

ROOT = Path('outputs/tennis_scene/visualize/player_view/blender-acceptance')
PACKAGE = ROOT / 'package'
OLD = ROOT / 'full/work/renders'
OUT = ROOT / 'full-gpu-denoise'
REF = '993c7e2310a76ebc7145ca33b9698be5b4996ddc'
OLD_REF = '2791ae2fc3fe660c01b79d337ef46b0752aa0fa3'
BLENDER = '/content/blender/blender-5.2.1-linux-x64/blender'
SOURCES = [16, 17, 19, 731, 735]
DEADLINE = datetime.fromisoformat('2026-11-08T20:19:37.670427+00:00')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def staging_path(path):
    staging = ROOT / "delivery-staging"
    staging.mkdir(parents=True, exist_ok=True)
    return staging / (Path(path).name + "." + uuid.uuid4().hex + ".partial")


def publish_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = staging_path(path)
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def context():
    info = json.loads((PACKAGE / 'manifest.json').read_text())
    sources = np.load(PACKAGE / 'source_frames.npy')
    assert info['num_frames'] == 1010 and np.array_equal(sources, np.arange(1010))
    assert [p['player_id'] for p in info['players']] == [0, 1]
    return dict(info=info, sha=sha(PACKAGE / 'manifest.json'), sources=sources,
                mesh=np.load(PACKAGE / 'player_smpl_valid.npy'),
                ball=np.load(PACKAGE / 'ball_3d_valid.npy'),
                positions=np.load(PACKAGE / 'ball_positions.npy'))


def validate(directory, identity, c, expected, *, complete, persistent=None):
    report = json.loads((directory / 'render.json').read_text())
    assert report['schema'] == 'tennis_scene_player_view_render_v1'
    assert report['player_id'] == identity and report['package_sha256'] == c['sha']
    assert report['engine'] == 'CYCLES' and report['blender_version'] == '5.2.1 LTS'
    assert report['gpu_devices'] and all(d['name'] == 'NVIDIA L4' and d['type'] == 'OPTIX' for d in report['gpu_devices'])
    assert not report['motion_blur'] and report['dither_intensity'] == 0
    if complete:
        assert report['complete'] is True
    if persistent is not None:
        assert report['persistent_data'] is persistent
        if persistent:
            assert report['denoising'] == {'enabled': True, 'denoiser': 'OPENIMAGEDENOISE', 'use_gpu': True}
    assert set(report['material']) == {'OpaqueGrayPlayer', 'OpaqueGreenBall'}
    assert all(m['alpha'] == 1 and m['emission_strength'] == 0 for m in report['material'].values())
    rows = report['frames']
    assert [r['source_frame'] for r in rows] == list(expected)
    with np.load(PACKAGE / f'camera_{identity}.npz') as f:
        camera = dict(f)
    size = tuple(c['info']['camera_config'][key] for key in ('width', 'height'))
    hashes = {}
    for output, row in enumerate(rows):
        index = row['source_frame']
        if persistent:
            assert np.isfinite(row['render_seconds']) and row['render_seconds'] > 0
        assert row['output_index'] == output and row['cache_index'] == index
        assert row['camera_valid'] == bool(camera['valid'][index])
        assert row['camera_reason'] == int(camera['reason'][index])
        hidden = [not (row['camera_valid'] and c['mesh'][p, index] and p != identity) for p in (0, 1)]
        assert row['player_hidden'] == hidden
        assert row['ball_hidden'] == (not (row['camera_valid'] and c['ball'][index]))
        if row['camera_valid']:
            matrix = np.eye(4)
            matrix[:3, :3] = camera['rotation'][index]
            matrix[:3, 3] = camera['location'][index]
            assert np.allclose(row['camera_matrix_world'], matrix, rtol=0, atol=2e-5)
        else:
            assert row['camera_matrix_world'] is None
        if c['ball'][index]:
            assert np.allclose(row['ball_position'], c['positions'][index], rtol=0, atol=1e-5)
        else:
            assert row['ball_position'] is None
        errors = np.asarray(row['pc2_max_error_m'])
        assert errors.shape == (2,) and np.isfinite(errors).all() and (errors <= 1e-5).all()
        image = directory / 'frames' / f'{output:05d}.png'
        with Image.open(image) as im:
            assert im.format == 'PNG' and im.mode == 'RGB' and im.size == size
            im.verify()
        hashes[image.name] = sha(image)
    return report, hashes


def archive_rows(path, directory, report, rows, indices, *, segment=None):
    assert rows and not path.exists()
    temp = staging_path(path)
    meta = {**report, 'frames': rows, 'complete': False, 'checkpoint_indices': indices}
    if segment is not None:
        meta['segment'] = segment
    prefix = f"player_{report['player_id']}"
    if segment is not None:
        prefix += f"/segment_{segment['source_start']:05d}"
    payload = json.dumps(meta, indent=2, allow_nan=False).encode()
    with tarfile.open(temp, 'w') as archive:
        header = tarfile.TarInfo(prefix + '/checkpoint.json')
        header.size = len(payload)
        archive.addfile(header, io.BytesIO(payload))
        for row in rows:
            name = f"{row['output_index']:05d}.png"
            archive.add(directory / 'frames' / name, arcname=prefix + '/frames/' + name, recursive=False)
    os.replace(temp, path)


def validate_prefix(c):
    players = []
    for identity in (0, 1):
        directory = OLD / f'player_{identity}'
        report_path = directory / 'render.json'
        if not report_path.exists():
            assert not list((directory / 'frames').glob('*.png'))
            players.append(dict(player_id=identity, count=0, directory=str(directory), report_sha256=None, png_sha256={}))
            continue
        count = len(json.loads(report_path.read_text())['frames'])
        report, hashes = validate(directory, identity, c, range(count), complete=False)
        covered = 0
        archives = sorted((ROOT / 'full/checkpoints').glob(f'player_{identity}_*.tar'))
        for path in archives:
            with tarfile.open(path) as archive:
                meta = json.load(archive.extractfile(archive.next()))
            start, stop = meta['checkpoint_indices']
            assert start == covered and stop <= count and meta['package_sha256'] == c['sha']
            covered = stop
        tail = None
        if covered < count:
            tail = ROOT / 'full/checkpoints' / f'player_{identity}_{covered:05d}_{count-1:05d}.tar'
            archive_rows(tail, directory, report, report['frames'][covered:count], [covered, count])
            # This closed old-prefix tail stays in the old checkpoint namespace.
            subprocess.run([sys.executable, '.agents/skills/tennis-drive/scripts/drive.py',
                            '--rclone-config', os.environ['RCLONE_CONFIG'], 'upload', str(tail), str(tail), '--verify'], check=True)
        players.append(dict(player_id=identity, count=count, directory=str(directory),
                            report_sha256=sha(report_path), png_sha256=hashes,
                            ignored_pngs_without_rows=sorted(p.name for p in (directory / 'frames').glob('*.png') if p.name not in hashes),
                            old_checkpoint_tail=str(tail) if tail else None, git_ref=OLD_REF))
    result = dict(schema='c3_audited_prefix_v1', package_sha256=c['sha'], players=players,
                  validated_at=datetime.now(timezone.utc).isoformat(), originals_modified=False)
    publish_json(ROOT / 'operations/prefix_manifest_gpu_denoise.json', result)
    return result


def blender_command(directory, identity, sources, *, debug=False):
    command = [BLENDER, '--background', '--factory-startup', '--python-exit-code', '1', '--threads', '4',
            '--python', 'src/tennis_scene/visualization/blender/shots/player_view.py', '--',
            '--package', str(PACKAGE), '--output', str(directory), '--player-id', str(identity),
            '--gpu-backend', 'OPTIX', '--source-frames', *map(str, sources)]
    if debug:
        command.insert(command.index('--python'), '--debug-cycles')
    return command


def perf():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == REF
    c = context()
    prefixes = validate_prefix(c)
    metrics = dict(git_ref=REF, package_sha256=c['sha'], source_frames=SOURCES, players=[], prefix_counts=[p['count'] for p in prefixes['players']])
    for identity in (0, 1):
        directory = ROOT / 'perf-gpu-denoise' / f'player_{identity}'
        subprocess.run(blender_command(directory, identity, SOURCES, debug=True), check=True)
        report, hashes = validate(directory, identity, c, SOURCES, complete=True, persistent=True)
        timestamps = [(directory / 'frames' / f'{i:05d}.png').stat().st_mtime_ns / 1e9 for i in range(len(SOURCES))]
        intervals = np.diff(timestamps).tolist()
        assert all(t > 0 for t in intervals)
        valid_mtime = [intervals[i-1] for i in range(1,len(SOURCES)) if report['frames'][i]['camera_valid']]
        valid_render = [r['render_seconds'] for r in report['frames'][1:] if r['camera_valid']]
        assert valid_mtime and valid_render
        worst = max(max(valid_mtime),max(valid_render))
        old = ROOT / 'preview-renders' / f'player_{identity}'
        reference = json.loads((old / 'render.json').read_text())
        assert reference['package_sha256'] == c['sha']
        references = {r['source_frame']: r['output_index'] for r in reference['frames']}
        images = []
        for i, row in enumerate(report['frames']):
            current = np.asarray(Image.open(directory / 'frames' / f'{i:05d}.png'), dtype=np.int16)
            prior = np.asarray(Image.open(old / 'frames' / f"{references[row['source_frame']]:05d}.png"), dtype=np.int16)
            mae = float(np.abs(current - prior).mean())
            span = int(current.max() - current.min())
            if not row['camera_valid']:
                assert span <= 4
            images.append(dict(source_frame=row['source_frame'], camera_valid=row['camera_valid'],
                               ball_hidden=row['ball_hidden'], player_hidden=row['player_hidden'],
                               reference_mean_abs_difference=mae, pixel_span=span))
        metrics['players'].append(dict(player_id=identity, intervals_seconds=intervals,
                                      mean_interval_seconds=float(np.mean(intervals)), max_interval_seconds=max(intervals), worst_valid_frame_seconds=worst, valid_mtime_seconds=valid_mtime, valid_render_seconds=valid_render, initial_render_seconds=report['frames'][0]['render_seconds'], denoising=report['denoising'],
                                      persistent_data=report['persistent_data'], gpu_devices=report['gpu_devices'],
                                      image_reference_match=all(r['reference_mean_abs_difference'] <= 3 for r in images), images=images))
    metrics['metadata_and_crc_passed'] = True
    metrics['image_reference_match'] = all(p['image_reference_match'] for p in metrics['players'])
    publish_json(ROOT / 'perf-gpu-denoise/perf_metrics.json', metrics)
    artifacts = ROOT / 'perf-gpu-denoise/artifacts'
    artifacts.mkdir()
    temp = staging_path(artifacts / 'perf-gpu-denoise.tar')
    with tarfile.open(temp, 'w') as archive:
        for identity in (0, 1):
            directory = ROOT / 'perf-gpu-denoise' / f'player_{identity}'
            archive.add(directory / 'render.json', arcname=f'player_{identity}/render.json')
            for image in sorted((directory / 'frames').glob('*.png')):
                archive.add(image, arcname=f'player_{identity}/frames/{image.name}')
        archive.add(ROOT / 'perf-gpu-denoise/perf_metrics.json', arcname='perf_metrics.json')
        archive.add(ROOT / 'operations/prefix_manifest_gpu_denoise.json', arcname='prefix_manifest.json')
    os.replace(temp, artifacts / 'perf-gpu-denoise.tar')
    print('C3_GPU_DENOISE_PERF_COMPLETE ' + json.dumps(metrics), flush=True)


def resume():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == REF
    assert not OUT.exists()
    c = context()
    prefixes = json.loads((ROOT / 'operations/prefix_manifest_gpu_denoise.json').read_text())
    gate = json.loads((ROOT / 'operations/performance_gate_gpu_denoise.json').read_text())
    assert prefixes['package_sha256'] == c['sha'] == gate['package_sha256'] and gate['passed'] is True
    original_prefixes = copy.deepcopy(prefixes)
    rates = {p['player_id']: p['worst_valid_frame_seconds'] for p in gate['players']}
    initial_budget = max(60.0, sum(p['initial_render_seconds'] for p in gate['players']))
    fresh_estimate = 1.2 * sum(1010 * rates[p] for p in (0,1)) + 180 + initial_budget
    prefix_estimate = 1.2 * sum((1010-p['count']) * rates[p['player_id']] for p in prefixes['players']) + 180 + initial_budget
    available = (DEADLINE - datetime.now(timezone.utc)).total_seconds() - 960
    if fresh_estimate < available:
        policy = 'fresh'
        for item in prefixes['players']:
            item.update(count=0, report_sha256=None, png_sha256={})
        estimate = fresh_estimate
    elif prefix_estimate < available:
        policy = 'reuse_audited_prefix'
        estimate = prefix_estimate
    else:
        raise RuntimeError(f'Neither fresh nor prefix fits: fresh={fresh_estimate}, prefix={prefix_estimate}, available={available}')
    print('C3_GPU_DENOISE_POLICY '+json.dumps({'policy':policy,'fresh_estimate':fresh_estimate,'prefix_estimate':prefix_estimate,'available':available}),flush=True)
    work, artifacts, checkpoints = (OUT / k for k in ('work', 'artifacts', 'checkpoints'))
    for directory in (work, artifacts, checkpoints):
        directory.mkdir(parents=True)
    publish_json(artifacts / 'run_start.json', dict(git_ref=REF, package_sha256=c['sha'], prefix_manifest=prefixes,
                 performance_gate=gate, forecast_seconds=estimate, available_seconds=available, selected_policy=policy, original_prefixes=original_prefixes, old_prefix_adopted=(policy=='reuse_audited_prefix'),
                 human_acceptance='pending', started_at=datetime.now(timezone.utc).isoformat()))
    blocks, provenance = [], []
    for prefix in prefixes['players']:
        identity, count = prefix['player_id'], prefix['count']
        old = Path(prefix['directory'])
        if count:
            assert sha(old / 'render.json') == prefix['report_sha256']
        directory = work / 'segments' / f'player_{identity}' / f'source_{count:05d}_01010'
        saved = 0
        if count < 1010:
            def snapshot(force=False):
                nonlocal saved
                p = directory / 'render.json'
                if not p.exists():
                    return
                report = json.loads(p.read_text())
                assert report['persistent_data'] is True
                n = len(report['frames'])
                if n <= saved or not (force or (saved == 0 and n >= 4) or n // 128 > saved // 128):
                    return
                rows = report['frames'][saved:n]
                assert [r['source_frame'] for r in rows] == list(range(count + saved, count + n))
                name = f'player_{identity}_source_{count+saved:05d}_{count+n-1:05d}.tar'
                segment = dict(source_start=count, source_stop=1010, git_ref=REF, report_path=str(p), output_index_scope='segment_local')
                archive_rows(checkpoints / name, directory, report, rows, [saved, n], segment=segment)
                blocks.append(dict(player_id=identity, path=name, segment_output_indices=[saved, n], source_frames=[r['source_frame'] for r in rows]))
                saved = n
                print('C3_GPU_DENOISE_CHECKPOINT_READY ' + json.dumps(blocks[-1]), flush=True)
            with subprocess.Popen(blender_command(directory, identity, range(count, 1010)), stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    if 'Saved:' in line or 'PLAYER_VIEW_COMPLETE' in line:
                        snapshot()
                code = process.wait()
                snapshot(force=True)
                if code:
                    raise RuntimeError(f'Resumed Blender ID{identity} failed: {code}')
            new_report, _ = validate(directory, identity, c, range(count, 1010), complete=True, persistent=True)
            assert saved == 1010 - count
        else:
            new_report = None
        old_report = json.loads((old / 'render.json').read_text()) if count else None
        assembled = work / 'assembled' / f'player_{identity}'
        (assembled / 'frames').mkdir(parents=True)
        rows, segments = [], []
        for kind, original, location, ref, offset in [('old_prefix', old_report, old, OLD_REF, 0), ('new_segment', new_report, directory, REF, count)]:
            if original is None:
                continue
            source_report = location / 'render.json'
            delivered = artifacts / f'render_{kind}_player_{identity}.json'
            temporary = work / (delivered.name + '.delivery')
            shutil.copyfile(source_report, temporary)
            os.replace(temporary, delivered)
            segment = dict(kind=kind, git_ref=ref, original_report_path=str(source_report), original_report_sha256=sha(source_report),
                           delivered_report=str(delivered), source_indices=[offset, offset + len(original['frames'])],
                           persistent_data=original.get('persistent_data', False), denoising=original.get('denoising', {'enabled':True,'denoiser':'OPENIMAGEDENOISE','use_gpu':False,'inferred_from_git_ref':OLD_REF}))
            segments.append(segment)
            for row in original['frames']:
                source = row['source_frame']
                assert source == len(rows) and row['cache_index'] == source
                source_png = location / 'frames' / f"{row['output_index']:05d}.png"
                if kind == 'old_prefix':
                    assert sha(source_png) == prefix['png_sha256'][source_png.name]
                shutil.copyfile(source_png, assembled / 'frames' / f'{source:05d}.png')
                delivered_row = copy.deepcopy(row)
                delivered_row['output_index'] = source
                rows.append(delivered_row)
        assert [r['source_frame'] for r in rows] == list(range(1010))
        report = {**(new_report or old_report), 'frames': rows, 'complete': True, 'segments': segments,
                  'persistent_data': None if count and new_report else (new_report or old_report).get('persistent_data', False),
                  'persistent_data_policy': 'per_segment', 'denoising': None if count and new_report else (new_report or old_report).get('denoising'), 'denoising_policy': 'per_segment'}
        publish_json(assembled / 'render.json', report)
        validate(assembled, identity, c, range(1010), complete=True)
        provenance.append(dict(player_id=identity, segments=segments))
    videos = work / 'videos'
    subprocess.run([sys.executable, '-m', 'src.tennis_scene.scripts.publish_player_views', '--package', str(PACKAGE),
                    '--renders', str(work / 'assembled'), '--output', str(videos)], check=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-n', '-i', str(videos / 'comparison.mp4'), '-frames:v', '240', '-an',
                    '-c:v', 'libx264', '-threads', '1', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                    str(videos / 'comparison_short.mp4')], check=True)
    from src.tennis_scene.visualization.player_view.video import verify_video
    short = verify_video(videos / 'comparison_short.mp4', frames=240, fps=c['info']['fps'], width=2560, height=720)
    publish_json(videos / 'short_comparison.json', dict(source_frames=[0, 240], verification=short, human_acceptance='pending'))
    for source in videos.iterdir():
        temp = work / (source.name + '.delivery')
        shutil.copyfile(source, temp)
        os.replace(temp, artifacts / source.name)
    publish_json(artifacts / 'checkpoint_manifest.json', dict(package_sha256=c['sha'], blocks=blocks, segments=provenance,
                 original_prefixes=original_prefixes, effective_prefixes=prefixes, selected_policy=policy, old_prefix_adopted=(policy=='reuse_audited_prefix'), git_ref=REF, human_acceptance='pending'))
    print('C3_GPU_DENOISE_FULL_DELIVERY_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['perf', 'full'])
    args = p.parse_args()
    if args.mode == 'perf':
        perf()
    else:
        resume()
