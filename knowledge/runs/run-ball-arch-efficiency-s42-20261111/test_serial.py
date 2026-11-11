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
