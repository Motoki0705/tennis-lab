# player_association

camera-local の person track を camera 間で同一人物どうし対応付けるタスク（#933）。
外観 Re-ID と足元のコート座標による幾何を統合し、[`cluster_multiview`](../../utils/README.md#matching)（camera 排他・推移律の MILP）で解く。
現在は評価（`evaluation/`）だけがある。方式の実装は後続の PR で追加する。

出力先の規約は[タスク出力規約](../OUTPUTS.md)を参照。

## 評価ラベル

実clipの camera 間対応の正解は **(camera, frame, box) → 人物** の形で持つ（`evaluation/labels.py` の `ClipLabels`）。
評価は任意の tracker の box を IoU で照合するので、tracker や tracking 設定を変えても同じラベルで評価できる。

| 役割 (`role`) | 意味 | 評価での扱い |
|---|---|---|
| `player` | rally の選手 | camera 間の同一性を評価する |
| `non_player` | それ以外で box が出たもの（隣コートの人、フェンス外の人、人でない誤検出） | 除外（`-1`）されるべき。camera 間の同一性は確認しておらず、採点しない |
| （`null`） | 2人を覆う box、または大半が背景の box | 評価から除く |

同じ camera・frame に同一人物の box が複数あってよい（tracker の重複 box や、画面端で途切れた体の一部）。

### 作成手順

1. `tests/benchmarks/player_association_clips.py` の `observe`（GPU、training queue 経由）で、clip ごとに人物検出・tracking を実行する。
   ラベル用の観測では `person_observations.max_tracks_per_camera=16` とし、対象外の人物も track として残す。
2. `--phase sheets` の track 一覧（等間隔 crop）と、tracklet の連結点・切り替わりが疑われる区間の密な crop、全体画像を目視し、
   track（必要なら frame 区間）ごとに人物を決めて review YAML に書く。
3. `--phase labels --review <yaml> --labels-dir <dir>` で box ラベルに変換する。review されていない track、存在しない track、
   区間の抜け・重なり、box を持たない人物はすべてエラーで停止する（検証から黙って落ちる box を作らない）。

ラベルに含まれるのは、観測 run の tracker が出した box だけである。tracker が一度も box を出さなかった人物は、ラベルにも無い。

### Meiji 3cam のラベル（v1）

review とラベルは [`tests/benchmarks/labels/player_association/meiji_3cam/`](../../../tests/benchmarks/labels/player_association/meiji_3cam/review.yaml) にある。
観測 run は `outputs/player_association/evaluate/meiji_clips/i933-observe-v1-20260927`（#933）。
clip ごとの選定理由と人物の説明は review YAML の `selection`・`people` を正とする。

- 3本の動画にまたがる4 clip: `video_000/clip_000`（人手の対応が既にある clip）、`video_000/clip_007`、`video_001/clip_001`、`video_002/clip_013`。
- 全 clip がシングルス。Meiji にはダブルスとボールボーイが無い。ダブルス・同色ウェアは合成データでのみ検証する。
- 本物の ID switch は `video_001/clip_001` cam0 の1件だけ。ID switch の検知は、この1件と合成データで評価する。
- 学習や擬似ラベルに使わない（test 専用）。

## 指標（`evaluation/metrics.py`）

予測は camera ごとの track `(D, T)` の box と、frame ごとの player ID（`-1` = 除外）。定義はモジュールの docstring を正とする。

| 指標 | 概要 |
|---|---|
| pair F1 | 同じ frame の、異なる camera の単位の組。正解 = 同一の選手、予測 = 同じ ID |
| group accuracy | frame 内のすべての単位が正しい frame の割合（対象外は除外され、選手ごとに ID が1つで、選手間で ID が異なる） |
| exclusion P/R | `-1` の precision/recall（選手の重複 box の `-1` は誤りに数えない） |
| ID switch P/R | 予測 track 上の、ラベル人物の変化（正解）と予測 ID の変化（予測）を ±`switch_tolerance` frame で1対1に照合 |

ラベルと照合できなかった予測 box は coverage として別に報告し、対応の誤りには数えない。
