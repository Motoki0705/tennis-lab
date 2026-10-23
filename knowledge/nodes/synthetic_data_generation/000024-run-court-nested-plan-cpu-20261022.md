---
id: run-court-nested-plan-cpu-20261022
type: run
task: synthetic_data_generation
sequence: 24
recorded_at: '2026-10-22'
title: B00〜B03の入れ子軌道計画と固定評価のCPU診断
provider: codex
issue: 1350
date: '2026-10-22'
status: done
config:
  device: cpu
  generation_seed: 20261022
  evaluation_seed: 77123
  base_trajectory_count_per_scene: 16
  maximum_trajectory_count_per_scene: 256
  density_levels: [0, 1, 2, 3, 4]
  base_step_m: 6.0
  position_guard_m: 0.5
  simulated_qa_acceptance_probability: 0.8
metrics:
  scenes: 4
  common_base_points: 672
  fixed_evaluation_plan_points: 92
  maximum_selected_train_per_series: 1536
  selected_train_union: 2976
  candidate_train_union: 20836
  portable_cpu_tests_passed: 8
repro:
  commit: 7e307f0499be3a2511afaba51966ae3362ec81b1
  command: 'SCENES_ROOT=<read-only-scenes> PROBE_OUTPUT_DIR=<new-output-dir> bash knowledge/runs/run-court-nested-plan-cpu-20261022/repro.sh'
artifacts:
  run_dir: knowledge/runs/run-court-nested-plan-cpu-20261022
  output_dir: tmp/court-nested-plan/run-02
parents: []
relations:
- to: run-court-sfm-all-scenes-drift-20260920
  rel: informed_by
papers: []
tags: [court, nested-sampling, trajectory, cpu, diagnostic]
---

## 結論と範囲

実際のB00〜B03のpublic scene exportとaccepted alignmentを入力に、軌道本数16〜256本・
密度段0〜4のCPU計画を作った。共通672点のID・K・pose・注視点・target binding・
v3投影教師が一致し、別namespaceの評価計画92点はtrain計画/seedを変えても不変だった。
候補の入れ子は作れる。しかし、品質棄却後の枚数・軌道数・分布は別に統制が要る。
**本runの設計提案は未合意で、新しい生成器・品質閾値を採用した結果ではない。**
案と判断依頼の正本は[設計文書](../../reports/court-scale-trajectory-design.md)。
本ノードは観測とその解釈を扱う。GPU描画・Colab・学習・精度測定は行っていない。
TensorBoard・収束曲線は対象外。

## 入力と旧方式の反例

入力のSHA-256は[metrics.json](../../runs/run-court-nested-plan-cpu-20261022/metrics.json)に記録した。
alignment・scene/cameras・保存config・既存dataset descriptor/indexの24ファイルは
処理前後で同じhashだった。旧形式のalignmentも厳格parserで読めた。
B01の保存configのみ、欠損key `sfm_complex_center_on_hull=false`を実験内コピーへ明示追加した。
資産変換の本実装や、旧保存設定そのものの変更は行っていない。
checkpoint本体のhash検証・可搬化は#1351の範囲。

| scene | 公開camera数 | court数 | 旧typed候補数 | 保存HFOV | hull中心指定 | 新候補Lのmin / median / max (m) |
|---|---:|---:|---:|---|---|---|
| B00 | 491 | 2 | 2187 | 75–110° | false | 24.75 / 65.32 / 82.66 |
| B01 | 266 | 3 | 2916 | 45–90° | 欠損、診断ではfalse | 25.01 / 59.83 / 117.82 |
| B02 | 157 | 1 | 2430 | 75–110° | true | 28.30 / 42.97 / 77.36 |
| B03 | 153 | 1 | 2430 | 75–110° | false | 20.70 / 30.29 / 70.10 |

旧候補数は予算選択前のtyped格子数で、物理的に異なる軌道数・採択数ではない。
この結果だけから16G本不足とは断定できない。近重複と採択率を調べて容量を判定する必要がある。
保存設定は全シーン5%のhull拡張で、新候補のprobeは全シーン共通の0%を使用した。

現行サンプラーで実sceneの軌道をstep1.05→0.525mにした反例を探索した。
coarse / fine / 共通点数はB00=32/56/8、B01=88/168/8、B02=56/104/8、B03=40/72/8。
半分のstepは2倍の点数にも包含にもならなかった。共通点は座標差1e-9m以内で照合した。
一方、完全に2倍・包含する軌道もあり、常に壊れるという観測ではない。
同じseedで16→32 groupへ増やした既存split関数では、旧16 groupのうち4/5/3/2 groupでsplitが変わった。
追加順とgroup indexに依存するjitter/viewの制約は設計文書のコード参照に示した。

## 入れ子と計数

実験の基準d=6m、各sceneの基準選択数24枚は、CPU検査を小さくするための値。
親の1500枚/sceneや本番のdの候補値へ流用しない。
一定の幾何predicateに加えてID hashで80%残す仮想QAを使った。
これは3DGSの実測採択率ではない。

| 段r | 各sceneの目標 | A1の候補軌道/scene | A2の固定候補軌道/scene | 各系列の選択数（全4scene） |
|---:|---:|---:|---:|---:|
| 0 | 24 | 16 | 16 | 96 |
| 1 | 48 | 32 | 16 | 192 |
| 2 | 96 | 64 | 16 | 384 |
| 3 | 192 | 128 | 16 | 768 |
| 4 | 384 | 256 | 16 | 1536 |

A1は新軌道block、A2は新密度birth層から追加した。両者の基準ID列を共有し、
全5段で前の選択を保持した。A2は基準で仮想QA採択が1点以上ある軌道を先に固定した。
B03ではそのpreflightで1familyを除外し、全段で実使用familyも16本を維持した。
A1は後段groupの非空を要求しないprobeなので、最終実使用はB00/B01/B02/B03で256/254/256/256本。
計画本数と採択後の実本数が異なる例であり、厳密な全G本使用の証拠にはしない。

| scene | 共通基準点（K/pose等の一致） | 固定評価計画点 | A1最大候補点 | A2最大候補点 | 仮想QA 100% / 90% / 70%時のA1最大幾何pass点 |
|---|---:|---:|---:|---:|---:|
| B00 | 192 | 24 | 3168 | 3072 | 2934 / 2644 / 2081 |
| B01 | 192 | 28 | 3100 | 3072 | 2709 / 2424 / 1885 |
| B02 | 156 | 20 | 2380 | 2496 | 2174 / 1938 / 1520 |
| B03 | 132 | 20 | 2108 | 2112 | 1683 / 1523 / 1209 |

長さの違いだけでもA1とA2の候補数がずれ、geometry/QAによる棄却でさらに差が広がる。
実際の全段の候補・eligible・選択数はmetricsの`counts`、長さ由来のずれは`length_count_drift`にある。
空の採択候補から1点を要求すると`insufficient accepted candidates: target=1, available=0`で停止した。
coarseに余剰があっても新density層が空なら補えない例も単体テストで確認した。

小規模Bは合計384枚とし、sceneごとの要求384/192/128/96をA1最大のID列から選べた。
1/2/3/4sceneで合計がすべて384になり、4scene版はA1段2と同じID。
B3の128枚は段2の96枚と段3の192枚の間にあり、完全な軌道prefixだけで表せない。
A1/A2最大の選択済み和は2976枚、候補和は20836点だった。前者を描画数の上限にできないことも分かる。

## guardが分布へ与えた影響

評価のval/testを各scene1軌道ずつ先に予約し、位置0.5mの保守的guardを使った。
曲線1024点の距離に弧長からの余裕を足したが、数値補間誤差を認証した実装ではない。
このguardが除外したfamily数はB00/B01/B02/B03で1347/923/787/1399、残したのは各256本。
それぞれ候補列の先頭から必要な本数が残るまでに調べた範囲の値で、共通の固定母数の率ではない。

| scene | circle / ellipse / rectangle / superellipse | complex / court |
|---|---|---|
| B00 | 132 / 44 / 39 / 41 | 174 / 82 |
| B01 | 129 / 50 / 32 / 45 | 134 / 122 |
| B02 | 64 / 62 / 72 / 58 | 125 / 131 |
| B03 | 85 / 47 / 66 / 58 | 226 / 30 |

生成前は各16本blockで形状・center kindを均衡化したが、guard後には大きく崩れた。
「安定順序だから分布も保つ」は反証された。出力層ごとのreservoir/不足判定と、
位置だけでなく視線・画角・投影も使うguardの比較を、未決の設計へ戻す。
seedを変えれば解消する、または狭い領域のsceneだけ例外を広げればよい、とは判断しない。

## 可視化と品質の限界

[3Dの計画軌道図](../../runs/run-court-nested-plan-cpu-20261022/geometry.png)は、
撮影カメラの位置と、新しい候補・予約評価の軌道を同じmetric frameで示す。
[保存済みRGBへの教師overlay](../../runs/run-court-nested-plan-cpu-20261022/saved-overlays.png)は、
対象courtのrenderer-visible KPが最少/最多の既存accepted画像を並べたもの。
B00の`court-sample-001772`、B01の`court-sample-000177`は対象が0/14点でもacceptedだった。
B02/B03の最少例は4/14、各sceneの最多例は14/14だった。
これは既存gateが全courtを合計する契約と整合する。新候補の画質を評価したものではない。
対象0点を負例として許容するかは未決で、観測だけで不良と断定しない。

動画はgitignore対象の`tmp/court-nested-plan/run-02/`に残した。
`B00-review.mp4`〜`B03-review.mp4`は新計画の位置・注視点・KP投影のCPU動画に、既存RGBを参照表示する。
`saved-B00-trajectory.mp4`〜`saved-B03-trajectory.mp4`は既存の1軌道を最大32枚抜き出し、
元RGBと教師overlayを並べて実際に視点移動させる動画。オレンジが対象court、緑が他court。
既存軌道のサンプル再生で、実撮影の時間/FPSや新規3DGSレンダリング動画ではない。
出典group/ID列・動画hashは[saved-video-review.json](../../runs/run-court-nested-plan-cpu-20261022/saved-video-review.json)。
動画・JPEGはknowledgeの保存方針に従ってgitへ入れず、再作成用のCPUコードと小さい図を残した。

目視した静止画には線のずれ・ネット周りのぼけ、対象courtが画面外にある例がある。
rendererとalignmentの誤差を分離した測定ではない。
新候補のα/depth・破綻・白線一致は未測定であり、閾値は4sceneの小規模描画後に判断する。
priorの地面高さ不整合診断も、独立した絶対3D正解には読み替えない。

## 再現・通常検証

base commitのsrcとpublic assets、bundle内の実験コードを用いる。Python 3.11.15、NumPy 2.3.5、SciPy 1.17.1。
`repro.sh`は読取専用のscene rootと新規の出力dirを要求する。入力root配下の出力・既存出力への上書きは拒否する。
実行した展開後コマンド・コードhashは[run.json](../../runs/run-court-nested-plan-cpu-20261022/run.json)。
検証記録は[validation.txt](../../runs/run-court-nested-plan-cpu-20261022/validation.txt)。

8つのCPUテストは非平面矩形の密度1/2/4段、点/乱数の順序独立、固定評価とfamily分離、
選択順序独立、不足、新density層不足、入力配下への出力拒否を検査し、全件passした。
ruff/mypyはknowledgeのPythonファイルにも手動適用した（repoのhook対象はsrc/tests/.spin）。
pre-commitは省略せずコミット時に実行する。

prototypeは点ごとのfield独立RNG、asset込みの本番ID、全joint層の均衡、
train内の全近重複排除、全軌道非空/coverage gate、streamメモリ、実QA/再開まで実装していない。
IDの一致はpinした資産内での検査。評価も採択後の画像集合ではなく幾何計画を固定したもの。
この制約を隠して、#1352以降が使う完成した生成器として扱わない。
