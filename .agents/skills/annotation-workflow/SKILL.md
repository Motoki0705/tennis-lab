---
name: annotation-workflow
description: Run tennis-lab annotation campaigns with pinned inputs, local image review, dataset publication, and additional detector learning. Use for planning, running, resuming, or reporting this workflow; consult the CLI README for currently connected task products.
---

# Annotation workflow

このskillは親controllerの操作を担当する。ラベル判断はtask adapterへ渡し、
ラベル規則や既存CLIを別スクリプトに複製しない。

## 読むもの

- [統合CLI](../../../src/tasks/annotation/README.md): 対応するtask/product、固定入力、再開・状態表示。
- [アーキテクチャ](../../../src/tasks/base/annotation/README.md): plan・採用・公開・学習の責務境界。
- 対象taskの [ball annotation](../../../src/tasks/ball_detection/annotation/README.md) / [player annotation](../../../src/tasks/player_detection/annotation/README.md): ラベル、推論、レビュー方針の正本。
- GPU実行時は [training-queue](../training-queue/SKILL.md)。実測結果を残すときは [knowledge-control](../knowledge-control/SKILL.md)。

## 計画と初期確認

既存のruntime、plan、resource scope、未完了attemptがあれば先に読む。継続依頼ではそれらを
引き継ぐ。新campaignを作ることと資源予算を新規に作り直すことを混同しない。
ユーザーが確定した件数、工程、素材条件、保存/PSS/GPU予算、採用条件を記録し、未回答の
主要な判断だけを質問する。実行済みの画像確認や実測を合成fixtureの結果で代用しない。

初めて使うモデル・代表素材はpreviewで定性確認する。未設定のレビュー条件や採用閾値を
ソースの既定値として埋め込まない。productionには確認済み方針と明示モデル参照を固定する。
CLIの `schema --type configuration` で入力契約を確認し、`configure` でplan・資源scope・
runtimeを固定する。継続campaignは `budget.action = reuse` で既存scopeを参照し、
code版・入力hash・工程costを結ぶ。
実行前に対象worktreeを固定し、実行中のコードを編集しない。
1サイクル全体の保存ピーク・GPU時間・空き容量を事前に見積もる。WSLではLinux内の空きと
仮想ディスクを置くhost volumeの空きを別に確認する。必要容量が確保されるまで運用を開始しない。
親controllerは指定されたtask/productの運用を担当し、実装不足は管理元へ報告する。
別taskの機能開発をサイクルの作業に追加しない。

## 実行と再開

履歴を登録してから新規候補を登録する。`run` / `tick` で共通台帳を進め、workerに必要な
画像・編集は親サービスが提供する。既存clipの再注釈、独立QA、旧remote入口を追加しない。
新入口で未対応のproductを、暗黙に除外してcampaign全体を完了させない。
動画の保持・削除は固定したpreparation policyに従う。JPEG保存方式では、元動画の不在を
推論・レビューの障害として再ダウンロードせず、prepared/retirement receiptを確認する。
詳細は[素材の保存契約](../../../src/tasks/base/annotation/collection/README.md#jpegを残して動画を削除する保存方式)を参照する。

停止時は `status --details` の現在attempt、理由、工程log、receiptを読む。予算超過なら
新規投入を止め、合意した対象rootすべての条件が回復するまで同じscopeのstopを保持する。
失敗を自動再試行しない。原因を修正した後、既存の実行権限内でCLIの明示再試行を使い、
理由と元attemptを残す。checkpoint再開と次campaignの新しい追加学習を区別する。

## 公開と報告

controllerのレビュー範囲照合と公開receiptを確認し、設定済みtaskの学習を進める。
追加教師がないtask、条件未設定、保留、失敗、不採用をそれぞれ表示する。
「注釈・dataset公開完了」と「学習・評価・採用完了」を分け、各receiptの参照と検証方法を
報告する。次campaignのモデルは採用台帳の確定版を参照し、最新checkpointを推測しない。

実動画・GPUで実行した範囲、合成試験だけの範囲、累積資源消費、残る判断と停止条件を明記する。

## 親controllerの定期監視

ユーザーが監視の停止を指示した場合は、その指示を優先する。サイクルの開始・再開だけを
理由に既存taskを再開・再登録しない。以下は監視が依頼されている間に適用する。

サイクルの開始・再開時は [codex-scheduled-followup](../codex-scheduled-followup/SKILL.md) を読み、
親controllerの作業を2時間（120分）ごとに監視する。ユーザーが別の間隔を明示した場合は
その指定を優先する。新規動画候補の選定・取得から注釈・dataset公開・累積学習・固定評価・
モデル採用までを対象とし、子workerだけを監視して親の継続判断を外さない。

親controllerをsubagentへ委任した場合は、委任指示にもこの監視責務を含める。
定期配信は稼働中の管理元CLIへ登録し、管理元がheartbeatごとに親controllerの進捗を確認して
許可済みの継続作業を委任する。subagentのthreadへ届くと仮定して別timerを作らない。
同じ対象の既存taskを再利用し、task directory・管理元thread・委任先・状態記録の場所を
サイクルの運用記録に残す。登録状態と次回実行を確認し、受信・完了の確認とは区別する。

各heartbeatで工程・子agent/job・成果物・資源状態を一度確認し、進捗と停止理由を報告する。
学習中は取得できるepoch/step・loss・評価指標も含める。監視による独立QAを追加しない。
次サイクルへの継続中も同じ監視taskを使い、認可済み作業全体の完了を検証したら
同skillの停止手順に従ってpauseし、次回実行がないことを確認する。
