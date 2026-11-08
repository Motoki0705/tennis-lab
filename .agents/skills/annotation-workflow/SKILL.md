---
name: annotation-workflow
description: Run tennis-lab annotation campaigns with pinned inputs, local image review, dataset publication, and additional detector learning. Use for planning, estimating, launching, monitoring, resuming, continuing stopped reviews, or reporting this workflow; consult the CLI README for currently connected task products.
---

# Annotation workflow

注釈campaignを運用する親の手順。ラベルの意味、CLIの入出力、状態の定義は各READMEを正本とし、
ここには判断の順序と、過去の運用で止まった箇所の避け方を置く。ラベル規則や既存CLIを
別スクリプトに複製しない。

## 読むもの

- [統合CLI](../../../src/tasks/annotation/README.md): 対応範囲、コマンド、再開・報告、レビューの進捗と継続。
- [アーキテクチャ](../../../src/tasks/base/annotation/README.md): plan・採用・公開・学習の責務境界。
- [ball annotation](../../../src/tasks/ball_detection/annotation/README.md) と
  [ball review](../../../src/tasks/ball_detection/annotation/review/README.md)、
  [player annotation](../../../src/tasks/player_detection/annotation/README.md): 予測・ラベル・レビュー方針の正本。
- GPUは[training-queue](../training-queue/SKILL.md)、実測の記録は[knowledge-control](../knowledge-control/SKILL.md)、
  長時間jobの起動と監視は[AGENTS.md](../../../AGENTS.md#学習の実行)に従う。

## 役割

運用親は、依頼の分解・見積もり・planの固定・起動・再開・報告を担当する。workflowに足りない
機能を見つけたら運用の途中で実装せず、確かめた事実と止まった工程を依頼元へ返す。
別taskの機能開発をサイクルの作業に混ぜない（#1088の運用で、運用親へplayer開発を任せた指示を
ユーザーが誤りとして取り消した）。

## 依頼を条件へ分解する

- sceneの`ball_detection.method=gpt_annotation`は、予測なしのローカル白紙注釈に対応する。
  [sceneの一連実行](../../../src/tennis_scene/pipeline/direct_annotation/README.ja.md)に従って
  元動画/frame/PTS・plan/予算・版付き探索方針・固定幾何・公開選択をrequestへ固定する。
  この経路では予測の定性確認を開始条件にせず、追加学習を起動しない。
  比較は`blank-register`で独立conditionを作り、そのruntimeを実行して`blank-compare`で計測する。
  比較runを通常datasetへ公開せず、合成試験と実GPTの認識受入を区別する。

- task/product、件数（channel×video×最大clip）、使用モデル、レビュー方針、学習・採用条件、
  資源予算に分ける。ユーザーが決めた値と未決定の主要な判断を分け、未決定のものだけを質問する。
  テストfixture・過去planの値やソースの既定値で埋めない。
- 依頼がCLI READMEの対応範囲（productionのtask/product、予測adapterが受け付けるcheckpoint、
  予測を初期値にするレビュー）に入るかを最初に確かめる。外れるなら実行前に報告する。
  対象を黙って減らす、別経路で代用する、未対応のproductを除いて完了扱いにすることをしない。
  既存clipの再注釈、独立QA、旧remote入口を足さない。
- 既存のruntime・plan・資源scope・未完了attemptがあれば先に読み、継続依頼ではそれを引き継ぐ。
  新しいcampaignを作ることと、資源予算を作り直すことを混同しない。

## 本番投入前の関門

本番の件数を投入する前に、次の結果をユーザーへ示して合意を得る。実データで測り、
合成fixtureの結果で代用しない。品質の確認は動画で見せる。

1. **予測の定性確認**: 撮影条件の異なる新規clipで予測を作り、予測を重ねた画像・動画で確かめる。
   球の不在を出せないモデルは非プレー・画面外のframeにも点を出し、workerがそこで球を探して時間を使う。
   2026-10-16のレビューが遅かった主因は、この点と約10 pxのずれだとユーザーが判断した（#1227）。
   レビュー方針（確認範囲・採用条件）はこの結果を見てユーザーが決め、版を付けてplanに固定する。
2. **レビュー費用の実測**: 決めたモデル・方針・worker条件で、小さなproduction campaign
   （1動画・数clip）のレビューを通す。`review-progress` の実測（worker 1時間あたりの検証済みframe、
   frameあたりtoken、clipあたりattempt数、途中停止の理由）から全件のworker時間とtokenを外挿し、
   件数・並列数・方針をユーザーと決める。2026-10-16は実測が停止後になり、1 clip（780 frame）に
   約90分、500 clipで約750 worker時間・入力約160億tokenかかると分かった。
3. **資源の見積もり**: 候補の実metadata（fps・解像度・長さ）から、元動画・保存JPEG・予測・
   レビュー画像とlog・telemetry・公開snapshot・checkpointの保存量、推論・学習・評価のGPU時間、
   PSSを見積もる。平均値や別taskの実績で外挿しない。WSLではLinux内の空きと、仮想ディスクを置く
   host volumeの空きを別に確かめる。全量を先に確保するか、資源scopeの停止条件の下で段階投入するかは
   ユーザーが決める。
4. **学習との整合**: 公開frameは元解像度で保存される。1080pのclipを含む学習窓は16 GiBのGPUで溢れ、
   720pのsnapshotを作り直した（#1231）。学習recipeの入力解像度・GPU memoryと候補の解像度が
   合わなければ、投入前に対処をユーザーと決める。JPEG保存方式は元動画を削除するため、
   後から別解像度で作り直すとJPEGの再圧縮が入る。

実データでの公開・追加学習・モデル採用は、まだ一度も完走していない。最初の本番は件数を小さくし、
公開と学習まで通すかをユーザーと決める。

## planを固定する

新規planの注釈worker条件は、`worker-decision` が返す値をそのまま使う（正本は
[決定記録](worker-decision.toml)、出典と決定日も確認する）。configureは不一致を補正せず拒否する。

```bash
.venv/bin/python -m src.tasks.annotation worker-decision
.venv/bin/python -m src.tasks.annotation schema --type configuration
```

- 残るplan条件をユーザーの決定と入力参照から構成し、`configure` でplan・資源scope・runtimeを固定する。
  継続campaignは `budget.action = reuse` で既存scopeを参照し、code版・入力hash・工程costを結ぶ。
- 既存campaignの再開では保存済みplanを使い、worker条件を含めて書き換えない。決定の変更は
  次に作るcampaignへ適用する。進行中campaignのplanをモデル変更のために書き換えない。
- ballのhybridモデルは[予測契約](../../../src/tasks/ball_detection/annotation/README.md#hybridの予測案)の
  明示schemaで登録し、checkpointの存在閾値を推論設定へ固定する。不在予測も未レビューであり、
  方針による画像確認と教師maskの検証を通す。
- reviewのtimeoutはworkerのpromptに時間上限として示される。関門2で測った1 clipの所要時間から決める。

## 起動と監視

- 実行前に対象worktreeとcommitを固定し、実行中のコードを編集しない。履歴を登録してから新規候補を登録する。
- 本体は `run --max-seconds N`。`complete`・`preview_ready`・`budget_stopped`・時間上限で終わる。
  判断が必要な停止があっても、稼働中のworker・工程が落ち着くまでserviceと回収を続け、
  そのあと `blocked` で終わる。判断待ちの間は、まだattemptのないclipのレビューを始めない。
- 長時間の `run` の起動と、異常終了・VM消失後の再開は[AGENTS.md](../../../AGENTS.md#学習の実行)の
  長時間ジョブの規則に従う。委任先は、最初のreview workerが稼働し、そのservice要求に応答が返るまでを
  見届けて終わる。以後の親は `run` プロセスの終了だけをイベントとして待ち、定期的に進捗を見に行かない。
  ユーザーが定期報告を指示した場合だけ、その指定に従う。
- `run` が終わったら `status`（必要なときだけ `status --details`）と `review-progress` を読み、
  次の節で判断する。`--max-seconds` は見積もった所要時間に合わせ、時間上限での終了も判断の機会とする。

## 停止から再開する

- 失敗を自動で再試行しない。原因を確かめ、理由と元attemptを残す明示の操作で再開する。
- **レビューの途中停止**: `review-progress` の `needs_decision` と各attemptの理由を読む。
  最新attemptが検証済みframeを増やしていれば、`continue-reviews --dry-run` で判定を確かめてから、
  `--reason` と、関門2の実測から決めた `--max-attempts` を付けて継続を記録する。
  増分がない停止、上限への到達、同じ理由の停止の繰り返しでは継続せず、理由・実測と選択肢
  （作業単位、方針の確認範囲、モデル）をユーザーへ示す。publicationは全clipの採用が揃うまで進まない。
- 基盤の不具合は、修正版をテスト・commitしてから、CLI READMEの「不具合修正後に同じcampaignを
  続ける」の手順で同じcampaignへ反映する。予算超過では新規投入を止め、合意した全rootの条件が
  回復するまで同じscopeのstopを保持する。
- 動画の保持・削除は固定したpreparation policyに従う。JPEG保存方式では、元動画の不在を
  推論・レビューの障害として再取得せず、prepared/retirement receiptを確かめる
  （[素材の保存契約](../../../src/tasks/base/annotation/collection/README.md#jpegを残して動画を削除する保存方式)）。
- checkpointからの再開と、次campaignの新しい追加学習を区別する。

## 公開・学習・報告

- controllerのレビュー範囲照合と公開receiptを確かめ、設定済みtaskの学習を進める。追加教師がない、
  条件が未設定、保留、失敗、不採用をそれぞれ表示する。成果物別のstatus・receipt・学習結果の意味は
  CLI READMEの「再開と報告」を正本とする。
- 「注釈・dataset公開の完了」と「学習・評価・採用の完了」を分けて報告する。次campaignのモデルは
  採用台帳の確定版を参照し、最新checkpointを推測しない。
- `report` と `review-progress` が返す不変参照を報告に添える。実動画・GPUで実行した範囲、合成試験だけの
  範囲、累積資源消費、レビューの実測（検証済みframe/時間・token）、残る判断と停止条件を明記し、
  実測はknowledge-controlでknowledgeに記録する。
