"""Single-writer, fail-closed T1 -> T2 -> C1 -> C2 -> C3 campaign.

Every durable generation is restored independently before advancing its pointer.
The same command resumes verified generations on a new, prepared Colab VM.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import torch
from protocol import CONFIG, DRIVE, INPUTS, PREPARED, REMOTE, ROOT, budget, model_config, read, recipe, write
from src.tasks.ball_detection.checkpoints.provenance import source_identity
from src.tasks.ball_detection.data.augmentation.video import AugmentationConfig
from src.tasks.ball_detection.training.runtime.execution import CoordinateRuntime
from src.tasks.ball_detection.training.runtime.images import jpeg_decoder_contract
from src.tasks.ball_detection.training.temporal.runner import train_temporal_model
from src.utils.artifact_store import RcloneArtifactStore, RcloneSnapshotStore
from src.utils.checksum import dual_sha256

CASES=('T1','T2','C1','C2','C3')
STAGES=[(case,stage) for case in CASES for stage in (range(2) if case.startswith('T') else range(1))]


def now(): return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())


def state_path(relative): return ROOT/'stage-status'/relative/'state.json'


def index_store():
    return RcloneArtifactStore(ROOT,remote='mounted_drive',remote_root=REMOTE,config_path=CONFIG)


def publish_index(path,value):
    write(path,value); index_store().publish_file(path)
    target=DRIVE/REMOTE/path.relative_to(ROOT)
    if dual_sha256(path)!=dual_sha256(target): raise RuntimeError('Mutable index readback differs')


def snapshot_store(directory):
    return RcloneSnapshotStore(directory,remote='mounted_drive',remote_root=REMOTE+'/'+directory.relative_to(ROOT).as_posix(),config_path=CONFIG)


def persistence(directory):
    store=snapshot_store(directory)
    relative=directory.relative_to(ROOT)
    def persist(path,event):
        if path!=directory: raise RuntimeError('Unexpected persistence path')
        names=['run.json','train_manifest.json','evaluation_manifest.json','train.jsonl']
        update=0
        if (path/'latest.json').exists():
            latest=read(path/'latest.json');update=latest['update']
            names+=['latest.json',latest['checkpoint'],f'weights-update-{update:08d}.pt']
            report=f'validation-update-{update:08d}.json'
            if (path/report).exists(): names += [report, report.replace('.json','.frames.jsonl')]
        names += [name for name in ('COMPLETED.json','STOPPED.json','FAILED.json') if (path/name).exists()]
        generation=f'{event}-{update:08d}-{uuid.uuid4().hex[:8]}'
        receipt=store.publish(generation,include=names)
        with tempfile.TemporaryDirectory(prefix='arch-readback-',dir=ROOT) as temporary:
            restored=RcloneSnapshotStore(Path(temporary)/'readback',remote='mounted_drive',
                                        remote_root=store.remote_root,config_path=CONFIG)
            verification=restored.restore(generation,manifest_sha256=receipt['manifest_sha256'])
        status=dict(stage=relative.as_posix(),event=event,update=update,receipt=receipt,
                    readback=verification,updated_at=now(),execution_commit=commit())
        publish_index(ROOT/'persistence'/relative/(generation+'.json'),status)
        publish_index(state_path(relative),status)
        print(json.dumps(dict(event='durable',**status)),flush=True)
    return persist


def commit(): return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()


def levels():
    decision=read(ROOT/'level-decision.json')
    if decision['status']!='confirmed': raise RuntimeError('MSDA levels are not confirmed')
    return tuple(decision['levels'])


def stage_name(case,stage,smoke):
    return Path('smoke')/case if smoke else Path(f'{case}-q{stage*6000:05d}-q{(stage+1)*6000:05d}' if case.startswith('T') else f'{case}-dpt-10000')


def restore_latest(relative):
    directory=ROOT/relative;remote_state=DRIVE/REMOTE/'stage-status'/relative/'state.json'
    if not remote_state.exists():
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError('Local stage exists without a verified Drive pointer; inspect it before recovery')
        return
    status=read(remote_state); receipt=status['receipt']
    if status['update']==0:
        if directory.exists(): directory.rename(directory.with_name(directory.name+'.abandoned-'+uuid.uuid4().hex[:8]))
        return
    if status['readback']['status']!='verified': raise RuntimeError('No verified generation')
    if directory.exists():
        latest=directory/'latest.json'
        if latest.exists() and read(latest)['update']==status['update'] and (directory/'COMPLETED.json').exists()==(status['event']=='completed'):
            expected=read(latest)
            if dual_sha256(directory/expected['checkpoint'])==expected['sha256']:
                return
        directory.rename(directory.with_name(directory.name+'.abandoned-'+uuid.uuid4().hex[:8]))
    snapshot_store(directory).restore(receipt['generation'],manifest_sha256=receipt['manifest_sha256'])


def prepare():
    if not DRIVE.is_dir() or not (DRIVE/'data/ball_detection').is_dir(): raise RuntimeError('Mounted project Drive required')
    if not CONFIG.exists():
        CONFIG.write_text('[mounted_drive]\ntype = alias\nremote = '+str(DRIVE)+'\n');CONFIG.chmod(0o600)
    verified=read(ROOT/'data-verified.json')
    if not verified['complete'] or verified['verified_files']!=2925 or verified['verified_bytes']!=157028755547:
        raise RuntimeError('Entire fixed input snapshot must be verified')
    c0=read(INPUTS/'common-dpt-T32/validation-update-00010000.json')
    evaluation=read(PREPARED/'evaluation_T32.json')
    for t in (32,64):
        manifest=read(PREPARED/f'evaluation_T{t}.json')
        if manifest['summaries']['val']['counts']['observed']!=85473: raise RuntimeError('Validation denominator differs')
    if c0['metrics']['support']['observed']!=85473: raise RuntimeError('C0 fixed evaluation is invalid')
    baseline=dict(mean_error_px=c0['metrics']['mean_error_px'],requires_reevaluation=(
        c0['manifest_sha256']!=dual_sha256(PREPARED/'evaluation_T32.json') or c0['image_decode']!=jpeg_decoder_contract('nvjpeg')))
    identity=source_identity()
    source={key:identity[key] for key in ('source_sha256','torch_version')}
    plan=dict(schema='ball_arch_campaign.v1',seed=42,levels=levels(),baseline=baseline,
              source=source,torch=str(torch.__version__),execution_commit=commit(),
              jobs=[dict(case=case,model=asdict(model_config(case,levels=levels())),
                         stages=[budget(case,stage=i).payload() for i in (range(2) if case.startswith('T') else range(1))]) for case in CASES])
    path=ROOT/'plan.json'
    if path.exists():
        plan['execution_commit']=read(path)['execution_commit']
        if read(path)!=json.loads(json.dumps(plan)): raise RuntimeError('Campaign plan changed; do not overwrite it')
    if not path.exists(): publish_index(path,plan)
    return plan


def validate_complete(output,bud):
    if not (output/'COMPLETED.json').exists(): return False
    latest=read(output/'latest.json')
    if not read(output/'COMPLETED.json')['complete'] or latest['update']!=bud.updates:
        raise RuntimeError('Completion marker/update is invalid')
    if dual_sha256(output/latest['checkpoint'])!=latest['sha256']: raise RuntimeError('Completed checkpoint is corrupt')
    report=read(output/f'validation-update-{bud.updates:08d}.json')
    if report['metrics']['support']['observed']!=85473: raise RuntimeError('Fixed validation denominator differs')
    pointer=read(DRIVE/REMOTE/'stage-status'/output.relative_to(ROOT)/'state.json')
    if pointer['event']!='completed' or pointer['update']!=bud.updates: raise RuntimeError('Completion is not durable')
    return True


def stage(case,number,smoke,stop):
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_name()!='NVIDIA L4': raise RuntimeError('Single L4 required')
    chosen=levels();relative=stage_name(case,number,smoke);output=ROOT/relative
    (ROOT/'locks').mkdir(parents=True,exist_ok=True)
    with (ROOT/'locks'/(relative.as_posix().replace('/','__')+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        restore_latest(relative)
        bud=budget(case,stage=number,smoke=smoke)
        if not smoke and validate_complete(output,bud): return
        options={}
        if (output/'latest.json').exists():
            options['resume']=output/read(output/'latest.json')['checkpoint']
        elif case.startswith('T'):
            if number:
                parent=ROOT/stage_name(case,0,False)
                if not validate_complete(parent,budget(case)): raise RuntimeError('Previous stage is incomplete')
                pointer=read(parent/'latest.json');checkpoint=parent/pointer['checkpoint'];parent_update=6000
            else:
                checkpoint=INPUTS/'common-dpt-T32/checkpoint-update-00010000.pt';parent_update=10000
            options.update(initial_artifact=checkpoint,initial_artifact_sha256=dual_sha256(checkpoint),parent_update=parent_update)
        origin=recipe(case)
        if smoke:
            origin['runtime']=dict(origin['runtime'],input_verification='lazy')
            print('SMOKE ONLY: CNN unfrozen; lazy per-shard checks after complete snapshot verification. Main recipe remains unchanged.',flush=True)
        train_temporal_model(architecture='query' if case.startswith('T') else 'dpt',
                             phase='finetune' if case.startswith('T') else 'pretrain',
                             manifest=PREPARED/f'train_T{64 if case.startswith("T") else 32}.json',
                             evaluation_manifest=PREPARED/f'evaluation_T{64 if case.startswith("T") else 32}.json',
                             model_config=model_config(case,levels=chosen),output=output,budget=bud,
                             runtime=CoordinateRuntime(**origin['runtime']),device=torch.device('cuda'),
                             augmentation=AugmentationConfig(**origin['augmentation']),distance_px=20.,
                             stop_after_update=stop,persist=persistence(output),**options)


def run_ordered(stages,finished,execute,save):
    for key in stages:
        if finished(key):
            save(key,'completed');continue
        save(key,'running')
        try:
            execute(key)
            if not finished(key): raise RuntimeError('Stage returned without durable completion')
        except BaseException:
            save(key,'failed');raise
        save(key,'completed')


def run():
    plan=prepare()
    with (ROOT/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (ROOT/'FAILED.json').exists(): raise RuntimeError('Explicit failure record requires inspection; no automatic skip/retry')
        status=dict(schema='ball_arch_serial_status.v1',pid=os.getpid(),updated_at=now(),commit=commit(),
                    stages={stage_name(c,s,False).as_posix():'queued' for c,s in STAGES})
        def finished(key):
            case,number=key; relative=stage_name(case,number,False);restore_latest(relative)
            return validate_complete(ROOT/relative,budget(case,stage=number))
        def save(key,value):
            status['stages'][stage_name(*key,False).as_posix()]=value;status['updated_at']=now()
            publish_index(ROOT/'serial-status.json',status)
        def execute(key):
            case,number=key
            command=[sys.executable,__file__,'stage','--case',case,'--stage',str(number)]
            logs=ROOT/'logs';logs.mkdir(exist_ok=True)
            with (logs/(stage_name(case,number,False).name+'.log')).open('ab',buffering=0) as stream:
                subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        try:
            run_ordered(STAGES,finished,execute,save)
            if plan['baseline']['requires_reevaluation']:
                subprocess.run([sys.executable,__file__,'evaluate-c0'],check=True)
            publish_index(ROOT/'COMPLETED.json',dict(complete=True,updated_at=now(),stages=status['stages']))
        except BaseException as error:
            write(ROOT/'FAILED.json',dict(error=str(error),traceback=traceback.format_exc(),updated_at=now()))
            try: index_store().publish_file(ROOT/'FAILED.json')
            except BaseException: traceback.print_exc()
            raise


def evaluate_c0():
    from src.tasks.ball_detection.model_io.mdd_temporal import restore_temporal_model
    from src.tasks.ball_detection.evaluation.temporal_heatmap import evaluate_temporal_heatmap
    from src.tasks.ball_detection.training.runtime.compilation import compile_coordinate_model
    checkpoint=INPUTS/'common-dpt-T32/checkpoint-update-00010000.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=restore_temporal_model(saved['model_artifact']).cuda()
    origin=recipe('C0');runtime=CoordinateRuntime(**origin['runtime']);runtime.configure(torch.device('cuda'))
    compile_coordinate_model(model,mode=runtime.compile_mode)
    output=ROOT/'C0-evaluation';output.mkdir(exist_ok=True)
    report=evaluate_temporal_heatmap(model,PREPARED/'evaluation_T32.json',split='val',runtime=runtime,
                                    device=torch.device('cuda'),artifact_sha256=saved['artifact_sha256'],distance_px=20.,
                                    output=output/'validation.json')
    if report['metrics']['support']['observed']!=85473: raise RuntimeError('C0 validation failed')
    snapshot=snapshot_store(output);receipt=snapshot.publish('completed-'+uuid.uuid4().hex[:8])
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
        RcloneSnapshotStore(Path(tmp)/'readback',remote='mounted_drive',remote_root=snapshot.remote_root,
                            config_path=CONFIG).restore(receipt['generation'],manifest_sha256=receipt['manifest_sha256'])
    publish_index(ROOT/'C0-evaluation-receipt.json',receipt)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','run','stage','smoke','evaluate-c0'))
    parser.add_argument('--case',choices=CASES);parser.add_argument('--stage',type=int,choices=(0,1),default=0)
    parser.add_argument('--stop',type=int)
    args=parser.parse_args()
    if args.mode=='prepare': prepare()
    elif args.mode=='run': run()
    elif args.mode=='evaluate-c0': evaluate_c0()
    else:
        if args.case is None: parser.error('--case required')
        stage(args.case,args.stage,args.mode=='smoke',args.stop)
