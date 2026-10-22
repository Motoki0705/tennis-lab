# 全フレーム共通区間のCPU再調査bundle

結論・数値は[knowledge node](../../nodes/ball_detection/000063-run-ball-all-frame-protocol-20261022.md)、
ユーザー確認待ちの規則は[protocol.md](protocol.md)を正本とする。
初回の中央16調査から独立した新run。入力・教師APIの分類関数は初回bundleを再利用し、
旧runのスクリプト・結果・protocolは書き換えていない。

## 再現

active checkout rootから実行する。`SOURCE_REPO_ROOT`は、`inputs.json`の元manifestと不変snapshot、
確認済みChat不在の元注釈を持つsource repoを指す。出力はactive checkout内の新しいディレクトリに限定する。
元manifestの絶対pathの可搬化は #1374 の担当で、別の入力を自動探索して代用しない。
元のTrackNet/Meiji CSVは既存の教師APIが要求しないため、この集計も要求しない。

```bash
SOURCE_REPO_ROOT=<source-repo-root> \
SURVEY_OUTPUT=tmp/ball-all-frame-protocol-repro \
bash knowledge/runs/run-ball-all-frame-protocol-20261022/repro.sh
```

manifestとsnapshot index/metadataの内容照合、旧窓の再生成、既存clipの意味的identity、
split間group非重複、旧runとのsnapshot/32窓の教師数一致を検査する。
Chat原注釈を既存APIで照合し、notes付き空frameをmaskへ戻す。
参照した元manifest/index/metadata・全1,800 RGB shard・1,120 Chat原注釈について、
inode/size/mtime/ctimeの不変を検査した。画像payloadのopen・復号は0。
すべてmetadata/indexだけのCPU処理で、GPU・Colab/Drive・hardlink操作はない。

`repro.sh`は出力の24証拠ファイルを記録版とbyte単位で比較する。
実時間・RSS・外側のGit版が変わる`execution.json`は比較から除く。
`execution.json`のcommitは読んだsrcの版であり、再現scriptは本bundleの版を使う。
APIが変わったcheckoutでは開始時に止まるため、該当するGit版で再現する。

## 後続へ渡す小さい入力

| file | 内容 |
|---|---|
| `common-intervals-<split>.jsonl` | `[snapshot_clip_index, [[start, stop_exclusive], ...]]`。PTSで分けた隣接区間も結合しない |
| `train-packs-<split>.jsonl` | `[snapshot_clip_index, [[first_start, last_start_inclusive, stride], ...]]`。採用済み128枚の組の開始列。strideは入力frame_stepではない |
| `absence-targets-<split>.jsonl` | `[snapshot_clip_index, [[start, stop_exclusive, class], ...]]`。class 2=reviewed_empty、3=out_of_frame。全適格不在の一意列 |
| `clip-catalog.csv` | snapshot indexとclip_id/split/source/group/native FPS/frame数の対応 |
| `assignment-rules.json` | 区間、学習分割、採用、抽出、評価owner、区間端、不在重複の機械向け規則案 |

`train-packs`の各行はclip_id順、開始位置は昇順。一様抽出の母集団はtrainの行をこの順で展開した全開始位置。
val/test版は支持数の対照計算用で学習しない。採点は`train-packs`による選別を使わず、共通区間を全て覆う。
入力の子窓に独立の再採用条件を課すと、更新全体の教師一致が壊れる。
共有するsample RNGはモデル初期化や拡張のRNGと分け、そのstateと消費位置を保存する。

不在集合は共通集合と全重複する。同じframeを2回集計せず、専用scopeは学習への追加samplerにしない。
正式manifest・配布storeへ変換するときはclip indexを黙って持ち越さず、clip_idで対応を検証する。
候補を正式採用する判断と、可変Tの常設実装はそれぞれ親・#1372の責任。

## 集計と照合用のartifact

| file | 内容 |
|---|---|
| `summary.json` | split×snapshot/旧32窓/共通/不在/重複/非重複の一意frame計数、coverage、集合identity |
| `source-fps.csv` | 上記scopeのsource、正確なnative FPS、source×FPS。支持数0の行も保持 |
| `groups-<scope>.csv` | `(source, group_id)`別の計数。共通/不在の両方で0支持のgroupも保持 |
| `budget.csv` | 各split・T・抽出案の窓数、1更新128枚時の期待教師数、全mask子窓数、位置教師なしの抽出単位の割合 |
| `edge-support.csv` | 共通frameの区間端距離別の教師支持数。T間で同一 |
| `evaluation-support.csv` | 評価窓数、全割当frame/教師数、MDD先頭になる教師数、owner割当の照合値 |
| `inputs.json` / `annotation-evidence.json` | 入力版・API版・検査結果、照合済みChat原注釈とnotes除外数 |
| `execution.json` | CPU時間、最大RSS、読んだsrc版、stat照合した入力ファイル数 |

coverageの分子は旧32窓の教師との**集合積**。`*_added_outside32`は旧窓外から追加された教師であり、
coverageに加えない。`intervals`はcommonでは条件(b)の区間数、その他scopeではframe maskの連続run数。
`budget.csv`の`sampling_units`はpaired案では128枚の組、independent案ではT枚窓。
`candidate_windows/accepted_windows`は重複を含む窓出現数で、unique starts数ではない。
`zero_teacher_child_windows`も出現数。`negative_only_packs`は位置教師0・不在教師1以上の抽出単位数で、
全128枚が不在である窓の数とは異なる。train以外の期待値は分布監査であって学習予算ではない。

集合identityは初回と同じ、clip_id→frame_indexの順のcompact JSONL
`[clip_id, frame_index, 1, exact_fps]`、教師identityはその末尾にclass codeを追加する。
classは0=mask、1=observed、2=reviewed_empty、3=out_of_frame。
owner identityはclip_id順・区間順のcompact JSONL`[clip_id,a,b,[owner_start_for_each_frame]]`。
内容照合値はJSON/CSVに保ち、文書本文には転記しない。

## 通常検証

pytestは共有ロックを明示し、指定された形式で実行する。

```bash
export PYTHONDONTWRITEBYTECODE=1
flock "$PYTEST_LOCK" .venv/bin/python -m pytest \
  knowledge/runs/run-ball-all-frame-protocol-20261022/test_all_frame_probe.py \
  knowledge/runs/run-ball-temporal-target-survey-20261022/test_temporal_target_probe.py -n 4 -q
.venv/bin/python -m ruff check knowledge/runs/run-ball-all-frame-protocol-20261022
MYPYPATH=knowledge/runs/run-ball-temporal-target-survey-20261022 \
  .venv/bin/python -m mypy --follow-imports=skip \
  knowledge/runs/run-ball-all-frame-protocol-20261022/all_frame_probe.py \
  knowledge/runs/run-ball-all-frame-protocol-20261022/test_all_frame_probe.py
```

synthetic timelineでreference/PTS境界・短区間・非整列末尾・同距離owner・全mask子窓を検査する。
保存済みの区間・不在列から一意集合を復号し、支持数と集合identityを再計算する。
元注釈のnotes照合と未確定教師の扱いは、再利用した初回runのテストも同時に実行する。
このrunはデータと規則のCPU調査で、学習曲線・精度測定・TensorBoardは対象外。
