# PR #2470: main の取り込みと、正式レビュー round 1 の指摘4・5への対応（その1）

#2449（#2444）、#2469（#2445）、#2461（#2459）はmainにマージ済み。main には #2610（SceneResult v3、ball event、`event_detection` の必須化）も入った。PR #2470 は main と多数のファイルで競合している。正式レビュー round 1（run parent-1739-1b51b8c3078245e4ab34994f4271b2cf、HEAD b62a0f3d）のmajor 5件のうち、このターンでは main の取り込みと指摘4・5に対応する。指摘1〜3は、#2462 の新HEADができた後に、次のターンで指示する。

## 1. main の取り込み
- `origin/main` を merge する（rebase はしない）。#2449・#2469・#2461 の内容は main 側を正とする。このbranchで手で取り込んだ版との差分は、main 側に寄せて解く。
- knowledge: このbranchの `000098`〜`000103` は、mainの `000098`〜`000100`・`000102`〜`000104`（`000101` は #2656）と同じrunの重複である。main 側を残し、重複を消す。このPRで独自にノードを足すなら、106 以降にする（105 は #2462 が使う）。

## 2. 指摘4: 新経路の公開後に、新旧比較ページと完了表を再生成できない
`publish` は `i2363-all-v1` の有効な `scene_assembly` を新経路へ切り替える。ところが比較CLI（mainの #2461）と `multiview_scene report` は、同じ有効indexを旧経路として読む。clip_000 から公開後のindexをメモリ上で組み立てると、比較側の `_read_scene` は `KeyError('alignment')` で止まった。完了表も `report.py:184–189` で、旧経路の `skeleton` と `alignment` の依存を要求する。
- 公開時に保存した旧参照（publish が記録する前の pointer）などから、比較元の世代を明示できるようにする。新経路を採用した状態で、比較CLIと完了表の両入口を検証するテストを付ける。黙って旧/新を推測する fallback は作らない。

## 3. 指摘5: #2610 の取り込み後に、保存済みバッチから新経路を起動できない
`multiview_scene/__main__.py:110–113` は、#2363 の `pipeline_config_yaml` を現行の `PipelineRuntimeConfig` に直接渡す。保存済み設定には `event_detection` と `execution.event_detection` がなく、#2610 は両方を必須にする。そのため `prepare`・`run` が設定の検証で止まる。
- 保存済み設定から、新経路が使う項目だけを明示的に読み取る契約を設ける。#2363 形式の fixture で `prepare`[local path]`run` の入口を検証する。SceneResult v3 を新経路の出力にするかどうかは、このターンでは決めない。現状の出力形式と、v3 にするなら何が要るかを報告に書く。

## 共通
- 関連テスト、pre-commit、knowledge 検査（`--base-ref origin/main`）を通して push し、PR本文を更新する。
- `i2363-all-v1/inputs/dataset` とベンチマークには書き込まない。`outputs/i2446-ballrev1` を消さない。GPU は使わない。

新HEAD（完全SHA）、解いた競合、指摘4・5の対処、テスト結果を報告し、CI待ちで終了する。
