# #2491 FastHMRの速度・定性品質の検証と別経路の実装（親 #2492）

担当issue #2491 の本文が仕様の正本。下の共通事項（親 #2492）が本文の「全69 clip」などより優先する。

## このissueで先に決めること

- **pipelineの別経路（recipe）の選び方の設計**は、このissueが先に固める。#2487 のECPose経路も同じ形で作るので、設計（既存の `src/tennis_scene/pipeline/definition.py` の標準recipe・component overrideの仕組みとの関係、cache identityへの反映、既定recipeを変えない保証）を早めに #2491 にコメントし、PRに含める。
- FastHMRの公式構成（HMR 2.0ベース・CameraHMRベース）の公開checkpoint・再現条件・時系列入力契約・TRT対応・SMPL/SMPL-Xの扱いを、実装前に公式コードで確認する（https://github.com/TaatiTeam/FastHMR 、論文 https://arxiv.org/abs/2510.10868 ）。公開されていない構成は根拠を残して実行不能とし、機構を削った独自モデルをFastHMRとして扱わない。

## 進め方

1. 公式構成の確認と経路設計を行い、要点を #2491 にコメントする。
2. FastHMRのwrapper・出力adapter（現行 `body_parameters` v2 と body_placement の契約へ）・別recipeを実装し、既定recipeの回帰テストを書く。GVHMRへの暗黙fallbackと、GVHMR成果とのcacheの混在をしない。
3. #2495 の固定入力manifestで、現行（HMR2特徴＋GVHMR）とFastHMRに同じ入力を与える。速度は #2496 harnessで測り、現行の基準値は #2496 の値を使う。
4. 代表6 clipで、元映像・各モデルの同期メッシュoverlayと、配置前後の3D比較を人間が見られる可視化を作る（人間の定性評価が品質判断の主になる。動画で見られるようにする）。
5. 結果をknowledgeに記録する。

PRは早めにdraftで作る。queueへ計測jobを `add` したらjob IDを宣言して終了してよい。

## 親 #2492 の共通事項（3子共通）

- 代表clip（2026-11-24 ユーザー合意）: video_000 の clip_003・clip_057・clip_004・clip_013・clip_040・clip_028。評価対象は各clipでGVHMRが実際に推論したframe（約30fps格子）×cam0〜cam3。起票時の「全69 clip」は母集団と読み替える。clip一覧・疑似GT・入力manifestは #2495 が作るものを、同じID・checksumで参照する（自分で作り直さない）。
- 速度・VRAMは #2496 の共通harnessと固定速度入力で測る。現行モデルの速度基準は #2496 の値を使う（同じ条件を測り直さない）。harnessに足りない機能があれば、自分で別のharnessを作らずに親へ返す。
- ライセンス同意やアカウントが必要な重み（RF-DETR-2XLのRoboflow PML 1.0、Sapiens2など）は、ユーザーが同意・用意したうえで評価目的に使うと合意済み。必要になったら、何に同意し何を用意してほしいか（URL・必要な環境変数名など）を `needs_parent` で親へ返す。自分で同意・登録しない。認証や利用許諾を回避しない。本番採用できるかはライセンス上の制約として記録する。
- Sapiens2-5BのPyTorch FP32はローカルGPUで試行し、OOMを実行不能として必要メモリの見積もりと記録する。別GPUでは測らない。
- GPUはtraining queueへ `add` して使う（queue workerは親側で稼働中。自分で `start`・`serve` しない）。ローカルGPUは RTX 5060 Ti 16GB。
- 新しいpipeline経路（ECPose・FastHMR）は既存経路を残した明示選択の別recipeとして作り、暗黙のfallbackやcacheの混在をしない。経路の選び方の設計は #2491 が先に固め、#2487 のECPose経路も同じ形に合わせる。
- #1739 の新経路コード（PR #2449・#2469・#2461・#2470）は2026-11-24時点で未マージ。下流の新経路へ通す確認でこのコードが要る場合は、その時点のマージ状況を親へ確認する。
- 公式重みが取得できない・ライセンスで使えない場合、別サイズ・別データの重みへ無断で置き換えない。

## 土台

worktreeの土台は、#2495 のPR #2498（head dfbacf50）と #2496 のPR #2497（head 8dbafd61）を統合したcommit 9455acef8。両PRはまだdraftで変わりうる。土台のPRが更新されたら必要に応じて取り込み、両PRのマージ後はmainへ載せ替える（親が指示する）。土台のPRのコード自体は変更しない。不具合や不足を見つけたら親へ返す。
