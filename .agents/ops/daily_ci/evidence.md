# 日次CI実行証拠（schema_version 1）

[controller](README.md) と同じcycle/requestを使う。collector #1497、回復判定 #1452 は
`PYTHONPATH=.agents/ops` で `daily_ci.evidence` をimportする。workflowはCPUの `Python` job、
knowledge、knowledge-webuiを維持し、GPUのdispatch-onlyとlabelerは変更しない。

## Artifactとidentity

jobごとに `daily-ci-<workflow>-<job>-<shard>-<run_id>-<attempt>-<request_id>` をuploadする。
workflowは `ci.yml` / `knowledge.yml`、jobはActionsのID、shardは現在 `all`。
`PLANS` が必須job/shard/検査IDの正本。`artifact_name(identity)` で期待名を生成できる。
新しいshardや検査構成はproducer/collectorを同時に変更する。

`manifest.json` のキーは追加なしで以下。

| キー | 内容 |
|---|---|
| schema_version | 整数1 |
| identity | repository, workflow, job, shard, cycle_id, generation, request_id, target_sha, checkout_sha, baseline_sha, deadline_at, run_id, attempt |
| started_at / finished_at | timezone付きISO日時。未完了のfinished_atはnull |
| checks | PLANS順の検査ID → outcome, started_at, finished_at, exit_code, argv |
| files | 同artifact内のファイル名 → 内容SHA-256。manifest自身は含めない |
| validation_error | 対象版/時刻等の拒否理由。正常はnull |

identityのgeneration/run_id/attemptは正整数、それ以外は文字列。repositoryはGitHubのowner/repo表記。
request_idはcontrollerと同じ `model.request_id`。SHAは完全な小文字40桁。
checkoutはtarget_shaを明示し、実HEAD、GITHUB_SHA、GITHUB_WORKFLOW_SHA、mainのref/workflow、
targetのorigin/main到達性、baseline→targetの祖先関係、JSTのcycle/期限を初期化・各検査前・終了時に検査する。
本番CLIに時計上書きはない。argv実行・環境変数経由で入力を扱い、GitHub入力をshellcodeへ埋め込まない。
controllerが初回/修正後generationを管理し、workflowが独自にcycleを作ったりbaselineを進めたりしない。

検査のoutcomeは `passed` / `failed` / `not_run` / `incomplete`。開始前はnot_run、開始記録後はincomplete、
プロセス終了値を取得したらpassed/failed。Lint等で前段が失敗した後のTestはnot_runのまま。
命令は固定argvで、期限までのtimeoutを設定する。最終時刻が09:00以降なら成功証拠を拒否する。
checkout/セットアップ失敗やhard killでは証拠を生成できない場合があり、欠落を成功と扱わない。

## JUnitとpytest

`checks.junit.xml` は全jobで各検査の実行状態を記録する。未実行/incompleteはskipped要素になるため、
その違いはmanifestのoutcomeで読む。pytestを実行したPython/knowledge jobはさらに以下を保存する。

- `pytest.junit.xml`: pytest自身が生成するJUnit。
- `pytest.events.jsonl`: masterのidentity、collection、実report、finishイベント。nodeidをJSON文字列としてそのまま保存。
- `pytest.json`: schema_version、同一identity、collected（選択されたnodeidの配列）、reports（nodeid→phase配列）、
  collection_errors、collection_consistent、finished、exit_code。

phaseは `when: setup|call|teardown`、`outcome: passed|failed|skipped`、`wasxfail: bool`。
xdistでもmasterが受信した実reportだけを記録し、workerのcollectionが同じことを照合する。
全phase成功・正常なcollection・終了確認のあるnodeidだけpassed。call成功でもteardown失敗ならfailed、
xfail/xpassやskipは回復としない。未実行/途中停止を区別し、改名・削除された旧nodeidの成功を推測しない。
parameter内の空白・`::`・括弧を分割しない。JUnitからnodeidを復元しない。
Web UIの各npm検査はchecksに記録し、実行できたPlaywrightのJUnitも保存する。

`always()`でfinalizeとuploadを試み、失敗時も保存可能な証拠を残す。job取消やartifact欠落・別attemptの証拠を
成功に合成しない。eventsだけ残る場合も部分証拠であり、完了manifest/JUnitの代用にはならない。

## 読取/検証API

```python
from daily_ci.evidence import PLANS, artifact_name, validate_bundle, validate_workflow, passed_nodeids

# expectedはartifactからコピーせず、保存cycle/requestとGitHubの対象run/attempt/jobから組み立てる。
verified = validate_bundle(extracted_directory, expected, remote_conclusion="success")
# {result: success|failure|incomplete, identity, checks, tests: {exact_nodeid: outcome}}
recovered = failure_nodeid in passed_nodeids(verified)
```

`validate_bundle` はschema・全identity・時刻・必須検査・ファイルdigest・JUnit・pytest完了情報を照合する。
remote_conclusionは**そのattemptのjob**の結論を渡す。cancelled/skipped/neutral/timed_outは拒否する。
読取/型/構造/XML/identity/欠落の例外はcollectorが取得・検証失敗として保持し、空の成功集合へ変換しない。
呼出元はGitHubのworkflow path/event/ref/run head SHA、run/attempt、artifact所属run、job一覧・必須shardを確認し、
`PLANS`の全jobが同じgeneration/対象版で成功した場合だけcontrollerへsuccessを返す。
`validate_workflow(directories, expected, conclusions)` はこの必須job/shard集合を一括検査する。
両mappingのキーは `(job, shard)`、expectedはidentityからjob/shardを除いたもの。
戻り値は `{result, jobs: {(job, shard): validate_bundleの戻り値}}`。不足/未知jobは例外になる。
一部jobが欠けた場合はcollectorがincomplete、実失敗はfailureとして扱う。取得障害や未実施を未復旧集合から除かない。
`passed_nodeids`は検証済みの戻り値にだけ使う。個別nodeがpassしていてもworkflow全体の成功を意味しない。

固定fixture: [pytest/xdist](../../../tests/e2e/agents_ops/fixtures/daily_ci/python/manifest.json)、[knowledge-webui](../../../tests/e2e/agents_ops/fixtures/daily_ci/knowledge-webui/manifest.json)。
使い捨てgit/実pytest/xdistの成功・失敗・skip・空白nodeid・改名・取消・欠落・SHA/attempt不一致・複数日baselineは
`tests/e2e/agents_ops/test_daily_ci_evidence.py`。fixtureは形式例で、実運用成功の証拠ではない。
切替と実CI/timer起動は#1497/#1452/#1498の接続検証後に親がまとめて行う。
