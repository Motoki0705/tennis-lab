"""Ordering/idempotence/failure contracts of this campaign, without a GPU job."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).parent))
import serial as campaign


def test_completed_stage_is_not_reexecuted_and_order_is_preserved():
    done={'T1-first'};calls=[];states={}
    def execute(key): calls.append(key);done.add(key)
    campaign.run_ordered(['T1-first','T1-second','T2-first'],lambda k:k in done,
                         execute,lambda k,v:states.update({k:v}))
    assert calls==['T1-second','T2-first']
    assert states=={'T1-first':'completed','T1-second':'completed','T2-first':'completed'}


def test_failure_stops_before_the_next_model_and_does_not_record_success():
    calls=[];states={}
    def execute(key): calls.append(key);raise RuntimeError('persistence failure')
    with pytest.raises(RuntimeError,match='persistence failure'):
        campaign.run_ordered(['T1','T2','C1'],lambda k:False,execute,lambda k,v:states.update({k:v}))
    assert calls==['T1'] and states=={'T1':'failed'}


def test_successful_process_without_a_verified_completion_is_failure():
    states={}
    with pytest.raises(RuntimeError,match='durable completion'):
        campaign.run_ordered(['T1','T2'],lambda k:False,lambda k:None,lambda k,v:states.update({k:v}))
    assert states=={'T1':'failed'}


def test_persist_callback_publishes_and_independently_restores_before_pointer(tmp_path, monkeypatch):
    import json
    import shutil
    if shutil.which('rclone') is None:pytest.skip('Requires rclone local-alias backend')
    root=tmp_path/'local';drive=tmp_path/'drive';drive.mkdir();root.mkdir()
    config=tmp_path/'rclone.conf';config.write_text('[mounted_drive]\ntype = alias\nremote = '+str(drive)+'\n');config.chmod(0o600)
    for key,value in [('ROOT',root),('DRIVE',drive),('REMOTE','campaign'),('CONFIG',config)]:
        monkeypatch.setattr(campaign,key,value)
    output=root/'T1';output.mkdir()
    for name in ('run.json','train_manifest.json','evaluation_manifest.json'):
        (output/name).write_text('{}\n')
    (output/'train.jsonl').write_text('')
    campaign.persistence(output)(output,'initialized')
    local=json.loads((root/'stage-status/T1/state.json').read_text())
    remote=json.loads((drive/'campaign/stage-status/T1/state.json').read_text())
    assert local==remote and local['event']=='initialized'
    assert local['readback']['status']=='verified' and local['readback']['files']==4
    assert (drive/'campaign/T1'/local['receipt']['generation']/'_SNAPSHOT.json').is_file()
