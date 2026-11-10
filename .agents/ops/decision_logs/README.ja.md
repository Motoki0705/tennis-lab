# 親の判断記録

[English](README.md) · [简体中文](README.zh-CN.md)

結果・親の判断・行動・訂正を表す共通契約version 1、非公開snapshotの保存、
公開用への変換を提供する。入口からの捕捉は #1815、Git公開と階層README生成は
#1816 の範囲。本パッケージはagent起動・Git/HTTP操作・定期実行・成果受入を行わない。

## 契約と責務

フィールド・列挙値・検証の正本はコードに置く。既存の
[workerログ](../worker_logs/README.md)の3ファイル形式は変えず、schemaの部品、
`Guard`、重複キーを拒否するJSON decoder、flock、永続化するatomic writerを直接再利用する。

| モジュール | 利用者との契約 |
|---|---|
| [schema.py](schema.py) | `new_id()`・`validate_record()`・`json_bytes()`。厳密なversion付きフィールドと欠測表現 |
| [graph.py](graph.py) | `validate_graph()`・`references()`・`states()`。型付き参照、DAG、行動履歴、明示的な検証 |
| [journal.py](journal.py) | `Journal`。原文の固定と準備・公開の追記履歴 |
| [projection.py](projection.py) | `render()`・`validate_public()`。公開単位 `{record, sanitization}` |
| [aggregation.py](aggregation.py) | `aggregate()`。すべての原記録を参照する表示用の集約 |

各記録は独立したUUID4 hexのID、種類、所属、区別した3つの時刻、型付きdataを持つ。
所属にはrepo・親・nullableな担当issue・活動の公開キー・既存workerの公開委任キー/turn・
nullableなreview公開キーを持つ。親の直接作業に架空の子issueは不要。
新規ID・キーは `new_id()` を使い、私的session/run IDや配送キーから導出しない。
対応はcapture entryの非公開contextだけに残す。

`validate_record()` は入れ子を含めキー・型を厳密に検査し、JSON入力では共通decoderで
重複キーを拒否する。`validate_graph()` は欠落・種類違い・別の親への参照、循環、
行動履歴の衝突、判断planとの不一致も拒否する。時刻の近さで因果を推定せず、
1結果から複数の判断・行動への分岐と、複数結果から1判断への合流を扱う。

## 結果・判断・実行状態

結果は機械観測・worker報告・親の検証を区別する。runtime終了、成果の品質、CI、
finalの有無、公開は別の観測。ファイルの存在で親の受領・承認を推定しない。
親の明示的検証は、検証対象を `verifies` で参照する結果として追加する。
`states()` はこの参照から未判断・確認未記録を導く。

結果は `previous_result_id` に対するコード・要件・具体指示・実行条件・確認済み範囲の
差を持つ。比較対象がなければ全項目を `unconfirmed` にする。
根拠には固定revisionか内容fingerprintを持たせる。編集可能なURLとfingerprintだけで
全文復元を保証しない。非公開・取得不能の根拠は私的URLを出さず、理由を残す。

判断は入力結果ID、短い理由、具体的な次の行動plan、再利用/再確認範囲、未解決事項、
再開条件と、設定・人間指示・親の選択を区別する出典を持つ。
初回判断は `basis` に要求・合意の固定参照を置き、先行結果を捏造しない。
planにも独立した公開IDを付け、分岐ごとに別のplan・行動を作る。

行動は判断とplanを参照する。判断IDが渡されなかった場合は両IDをnullにし、
`decision_missing_reason` を必須にして判断を作らない。
状態の更新は `previous_action_id` を持つ新たな行動snapshotで表し、所属・判断・plan・
操作種類を保持する。結果は先行する要求済み/実行中の行動snapshotを参照し、
その後の終了snapshotが `result_ids` で結果を参照する。同じsnapshotを双方から
参照すると循環になる。`decided_not_executed` の行動から結果は作れない。

訂正は対象と理由を持つ追記。必要なら同じ種類・所属の別の置換記録を参照する。
元記録は残り、graphは暗黙に置換したり、訂正を新たな観測へ読み替えたりしない。

## 欠測・時刻・使用量

取得状況は `{status: observed, value: ...}` または
`{status: unavailable, reason: ...}`。不明な時刻・時間・使用量・実効設定を、
0・選択モデル・現在時刻で補わない。観測した0は有効。
発生/受信時刻は取得状況を持ち、`recorded_at` は実際の記録時刻。
観測した時刻はすべてtimezoneを必須にする。

選択model/effortは指定元を保持し、実効値は独立して観測する。計測値は根拠を持つ。
使用量はturn単位/累積の区別と公開counterキーを持つ。
集約・本パッケージは累積値を加算せず、金額を推定せず、cache分を課金量へ読み替えない。

最終報告欠測は `missing_final` などの理由を残す。未判断、親の確認未記録、
判断済み未実行、起動前拒否、公開待ちは別のフィールド/状態として保持し、
runtimeの終了値から推定しない。

## 非公開journalと公開境界

既存の[parent state](../parent/README.md)を使い、
`Journal(parent_store.directory, repo, parent)` を作る。
`decision-logs/` 配下にrepo・親を分離した親ごとのjournalを置く。
ディレクトリはprivate、ファイル/lockは所有者のみアクセス可とし、symlinkを拒否する。
entry配列は追加だけ。flock・ファイル/ディレクトリfsync・atomic replaceで追記する。
破損JSON、重複/未知キー、未知version、所属の変化を空状態に置き換えない。
保証範囲はローカルfilesystemで、分散した担当排他ではない。

- `capture(key, record, private=...)` は入力を固定する。非公開contextはsession、run、
  その他のID、checkout root、nullableなcheckout commitを持つ。
  記録IDを省略するとlock内でUUID4を割り当て、同じprivate配送キーの再送では
  同じIDを返して書き込まない。入力変更は拒否し、呼出側は時刻・所属も固定版を再利用する。
- `prepare(record_id, key=..., at=...)` は安全な公開単位を固定するか、原文を残して
  `blocked` を追記する。依存先から準備する。blockedの再検査は `retry=True` と
  新しい配送キーで行い、元の作業は再実行しない。readyになった変換は置換できない。
- `publication(record_id, key=..., status=..., at=..., evidence=..., error=...)`
  は捕捉と独立してpending・blocked・publishedを追記する。
  publishedは固定した公開根拠を必須にし、終端状態とする。
  Git/HTTPの実際の保存確認は呼出側の責任で、本libraryの認定ではない。
- `public_payloads()` は依存順の検査済み公開単位だけを無害化件数付きで返す。
  公開側はこの単位を使って再検査する。`public_records()` はgraph/集約用の記録だけの表示。
  `status()` は非公開対応を出さず、捕捉・公開・欠測の状態を返す。

共通のworkerログguardで文章を無害化し、全公開fieldと無害化metadataを再検査する。
既知の私的IDとlocal pathを件数付きでマスクする。
固定参照URL・ID・モデルmetadataは意味を書き換えず検査する。
残った資格情報、未知UUID/session表記、危険なURL、制御文字は準備を拒否する。
未知の秘密を完全に分類する機構ではない。Git path・branch・生成README・PR本文は
公開側でも公開検査する必要がある。

## 表示用の集約

`aggregate(records)` は、対象・所属・状態・行動・条件キー・モデル設定が同じで、
比較参照が連鎖し、全変更が明示的に変更なしである隣接した機械確認/取得再試行を集約する。
捕捉側は同じ条件にだけ同じランダムな条件キーを使う。
未確認/変更あり、AI起動/継続、判断、行動snapshot、訂正は境界として残す。
1つの取得行動内の試行は集約でき、別の行動は表示を残す。
後の判断・検証・訂正が参照する途中の観測も、参照が系列の後に現れる場合を含めて表示を残す。
全記録ID・件数・初回/最終記録時刻・最終記録IDを返し、利用側は最終記録から結果を引く。
原記録と途中の失敗は残り、時間や使用量は補わない。

## 検証

```bash
.venv/bin/python -m pytest tests/unit/agents_ops/test_decision_logs.py \
  tests/integration/agents_ops/test_decision_log_journal.py \
  tests/unit/agents_ops/test_worker_logs.py -n 1 --durations=10
```

分岐/合流、明示した欠測、負例、無害化、集約、実際の2process追記、fsync中断、
破損journalを使い捨てstateで検証する。GPU・agent・networkは不要。
実運用の捕捉/公開とmerge後の受入確認は後続タスクに残る。
