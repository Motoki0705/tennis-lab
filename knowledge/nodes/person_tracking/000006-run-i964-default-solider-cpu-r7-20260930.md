---
id: run-i964-default-solider-cpu-r7-20260930
type: run
task: person_tracking
sequence: 6
recorded_at: '2026-09-30'
date: '2026-09-30'
title: COCO既定接続とSOLIDER推論portのCPU整合確認
issue: 964
provider: codex
status: done
config:
  source: COCO full-frame 0.30, 800/1333
  device: cpu
  encoder: solider_swin_base_msmt17
  semantic_weight: 0.2
  feature: normalized pre-BN 1024-d
metrics:
  cpu_parity_crops: 2
  max_abs_difference: 0.0
artifacts:
  run_dir: knowledge/runs/run-i964-default-solider-cpu-r7-20260930
parents:
- run-i964-fullframe-selection-r6-20260930
relations: []
papers: []
tags: []
---

## 確定判断とpipeline接続

[2026-09-30ユーザー判断](https://github.com/Motoki0705/tennis-lab/issues/964#issuecomment-5900983788)を
`2aaa42a3`でpipelineへ接続した。COCO全画面0.30、800/1333を既定にし、#937は明示設定で残した。
新しい`player_selection`はrun 6の規則を再利用する。core・連結・distinct滞在・全観測保持・capの定義は
[実装docstring](../../../src/tasks/person_tracking/court_linking.py)を正本とする。
raw trackingのROI/累計6制限とLab事前連結は除去し、group上限だけを選別後にかける。
`selected`は全元観測を保持し、v3へ渡す1 group/frame表のhandoff重複縮約は`origin_rows`に残す。
run 6のCOCO参照への偏り、4 singlesのみ、wide20欠落という制約は解消したと扱わない。

## SOLIDERのCPU確認

[上流SOLIDER-REID](https://github.com/tinyvision/SOLIDER-REID/tree/8c08e1c3255e8e1e51e006bf189e52cc57b009ed)の
Swin推論graphだけを移植し、MIT表示を保持した。MMCV等の学習環境は導入しない。
384×128、mean/std=.5、semantic weight=.2、BN前特徴をL2正規化する公開MSMT17 recipeを固定。
使用重みSHA-256は`81555144f412d46182d9cc8a0a01334f470a3484ce2fede88af9a5779d2a05a7`。
classifier/BN以外は全tensorをstrictに読み込み、別重みの推定や欠損補完をしない。

`video_000/clip_000` cam0 frame0のCOCO .30先頭2 crop（近側/小人物）で、
上流forward（semantic conditionをCPU tensorで明示、未使用MMCV load importだけ除外）とportの特徴は最大絶対差0、norm=1だった。
[入力box・動画hash・結果](../../runs/run-i964-default-solider-cpu-r7-20260930/solider_cpu_parity.json)、
[code hash](../../runs/run-i964-default-solider-cpu-r7-20260930/code_hashes.json)に実測の範囲を固定。
全体CIで上流forward内のassert/type確認/getattrがrepoのcomputation-only規則に抵触したため、
固定入力の検査をtyped wrapperに集約し、norm参照をconstructorで束縛した。検査の除外範囲や許可リストは変更していない。
修正後も同じ2 cropを再実行し、上流とcheckpoint key集合が一致、特徴の最大絶対差0を確認した。
これは2 cropの推論整合だけで、追跡精度、encoder採用、GPU性能の証拠ではない。学習曲線は無い。

## 次の実行

[特徴抽出plan](../../runs/run-i964-default-solider-cpu-r7-20260930/feature_plan.json)は、4 dev×3camera、10,491 frame、40,531検出をCPUでhash検査済み。
ViTPose/CLIP/SOLIDER全重みと検出元のhashを固定した。ViTPoseは一度だけ計算し、SOLIDERは保存poseを再利用する。
rawの全人物rowを保持し、cropが画像内で正面積にならない場合は外観maskに欠測を明記する。
KPRは推論portが未完成のため対象外。GPU抽出と2–3 tracker/encoder比較、#933再評価、全pipeline、未見一回評価は未実施。
次runでqueue結果を回収し、方式比較へ進む。未見の画像・ラベルを調整に使っていない。
