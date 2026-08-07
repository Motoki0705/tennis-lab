---
{
  "id": "proposal-plcs-global-tracklet-relinking",
  "type": "proposal",
  "title": "PLCS推論後へcourt-space global tracklet relinkingを追加する",
  "curator": "chatgpt-schedule",
  "date": "2026-08-07",
  "status": "candidate",
  "issue": 711,
  "task": "multi_object_tracking",
  "repo_paths": [
    "src/tasks/plcs/inference/tracking_predictor.py",
    "src/tasks/plcs/training/tracking_matching.py",
    "src/tasks/base/training/tracking_metrics.py"
  ],
  "hypothesis": {
    "statement": "学習済みPLCSを固定し、時間的に非重複なpresence fragmentだけをcourt-space運動整合性でoffline再接続すると、position/presence精度を維持したままID switchを減らせる",
    "expected_effect": "3 test seeds平均でid_switchesを20%以上削減し、position_error悪化1%以内、presence_f1低下0.5 percentage point以内に保つ",
    "failure_condition": "illegal overlapが1件以上、duplicateまたはinactive false positiveが10%以上増加、後処理がbaseline inference時間の10%を超える、または3 seeds中2つ以上でid_switchesが減らない"
  },
  "evaluation": {
    "metrics": [
      "id_switches",
      "position_error",
      "presence_f1",
      "duplicate_active_tracks",
      "inactive_query_false_positives",
      "illegal_overlap_count",
      "postprocess_time_ms"
    ],
    "baseline_nodes": [],
    "seeds": 3,
    "acceptance": "固定checkpoint・presence threshold・test scenesで3 seeds平均id_switchesを20%以上削減し、position_error悪化<=1%、presence_f1低下<=0.5 point、illegal_overlap_count=0、後処理時間<=baseline inference時間の10%を満たす"
  },
  "evidence_runs": [],
  "parents": [],
  "relations": [
    {
      "to": "paper-doi-10-1145-3728423-3759416",
      "rel": "derived-from"
    }
  ],
  "tags": [
    "literature",
    "plcs",
    "multi-object-tracking",
    "tracklet-association"
  ]
}
---

## 背景

PLCS track-query modelはclip内identityをquery slotで維持するが、遮蔽やpresence断裂後に別fragmentとして現れたtrackを推論後にglobal再接続する段階を持たない。GTATrack/GTAのlocal-to-global設計を、外部ReID依存を持ち込まずcourt-space出力に合わせて最小化する。

## 現行実装との差分

`tracking_predictor.py`の出力を時間的に非重複なpresence segmentへ分割し、終端position、短時間velocity extrapolation、rotation差から候補costを作るoffline connectorを任意で追加する。model weight、loss、presence threshold、query output contractは変更しない。

## 最小検証

固定checkpointと同一generated test scenesでbaseline出力を保存し、connector有無だけをA/Bする。時間的に重なるsegmentのmergeは禁止し、appearance ReID embeddingは初回実験に追加しない。

## 比較対象

正式baseline runが未登録のため`status: candidate`とする。現在利用するPLCS tracking baselineをformal graphへ登録してから`ready`へ進める。

## 合格条件と停止条件

frontmatterのacceptanceを満たす場合だけproduction設計へ進む。illegal overlap、false positive増加、計算量超過、seed間で一貫しないID改善があれば棄却する。

## リスク

原論文は多人数fisheye soccerとappearance ReIDを前提とする。geometry-only relinkingは近接・交差trajectoryで誤mergeする可能性がある。公式pipelineの依存をvendorせず、tennis-labの既存metricとcourt-space表現だけで独立実装する。
