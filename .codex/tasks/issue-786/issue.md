# GitHub Issue #786

- URL: https://github.com/Motoki0705/tennis-lab/issues/786
- State: OPEN
- Upstream updated at: 2026-08-23T03:23:16Z
- Snapshot SHA-256: `6279b189d4b3c0a7c11da3e605fbc252624f5a60ec808db2c476e061f55fa6a9`
- Acceptance checklist SHA-256: `95bcebf4388fdba9773e3c538c9e22caf82b6e4a413ec1241e9a58b0c4483032`
- Acceptance checklist item count: 22
- Labels: enhancement, ci, tests, module: base, module: blcs, module: plcs, module: synthetic-data, module: tennis-scene

## Acceptance checklist

- AC-001: versioned contractの単一正本が、`v1=(5.485,11.885,1.07)m`、`v2=(11.885,11.885,11.885)m`を返し、未知versionを明示的エラーにする。 (source checkbox: unchecked)
- AC-002: Hydraの共通configからBLCS / PLCS / SLCSの生成・学習・評価・推論へ同じnormalization versionが伝播し、`v1` / `v2`を明示的に切り替えられる。 (source checkbox: unchecked)
- AC-003: 初回導入時のdefaultが`v1`であり、既存config・dataset・checkpointを用いた代表的な推論、metric、lossの数値が変更前と許容誤差内で一致する。 (source checkbox: unchecked)
- AC-004: metadataを持たない既存dataset / checkpointは`v1` runtimeでのみ利用でき、`v2` runtimeでは明示的エラーになる。 (source checkbox: unchecked)
- AC-005: runtime config、dataset metadata、checkpoint metadataのversionまたは`scale_xyz`が一致しない場合、resume・evaluation・inferenceが明示的エラーになる。 (source checkbox: unchecked)
- AC-006: 任意shape`(...,3)`の物理positionについて、BLCS / PLCS / SLCSの各versionの`normalize -> denormalize`が最大絶対誤差`1e-5m`以下で元の値を復元する。 (source checkbox: unchecked)
- AC-007: `v2`正規化空間でdoubles sideline、baseline、net postがそれぞれ`x≈±0.4615`, `y=±1`, `z≈0.0900`となり、物理コート寸法は変更されない。 (source checkbox: unchecked)
- AC-008: BLCSの`ball_pos_norm`生成、position / velocity decode、differentiable projection、standard / tracking metricが選択versionの同じscale contractを使う。 (source checkbox: unchecked)
- AC-009: BLCS gravity priorのnormalized second differenceが選択versionのZ scaleに対する`-g*dt^2/scale_z`と一致し、固定値を使うtracking gravity targetもversionごとの整合が検証されている。 (source checkbox: unchecked)
- AC-010: PLCSのposition生成・target復元、standard / tracking predictor、metric、canonical-to-world統合、3D / top-down rendererが選択versionの同じscale contractを使う。 (source checkbox: unchecked)
- AC-011: PLCSの`canonical_pose_3d`などroot-relative meter座標は再スケールされず、position translationだけがversioned normalized contractを使う。 (source checkbox: unchecked)
- AC-012: SLCSのposition、evaluation、metric、adapter、inference、uncertaintyのnormalized↔meter変換が選択versionへ追随し、統合`SceneResult`は引き続きcourt/world`[m]`を返す。 (source checkbox: unchecked)
- AC-013: `v2`で同じ物理position誤差をX/Y/Zへ個別に与えたとき、defaultのunweighted Smooth L1 lossが全軸で一致し、共通の物理Huber遷移点が設定・テスト・ドキュメントで確認できる。 (source checkbox: unchecked)
- AC-014: `v2`のBLCS / PLCS default position lossとHungarian position costに、旧axis-scale補正由来の非等方weightが残っていない。`v1`の既存config挙動は維持される。 (source checkbox: unchecked)
- AC-015: 新規生成dataset metadataにnormalization version、`scale_xyz`、position / velocity単位が保存され、root / scene間のmissing・unknown・mixed contractをloaderが拒否する。 (source checkbox: unchecked)
- AC-016: 新規checkpoint metadataにnormalization versionと`scale_xyz`が保存され、checkpoint由来のversionが推論時に復元・検証される。 (source checkbox: unchecked)
- AC-017: 既存`v1` dataset / checkpointは上書きされず、`v2` dataset / checkpointとartifact名・metadataの両方で識別できる。 (source checkbox: unchecked)
- AC-018: BLCS / PLCSの`v2` datasetが別versionとして生成され、保存normalized値をmetersへ戻した値が生成時world値と最大絶対誤差`1e-5m`以下で一致する。 (source checkbox: unchecked)
- AC-019: `v2`のBLCS / PLCS baselineを再学習し、`v1` baselineとの比較を物理m単位の軸別metricと統合metricで記録する。 (source checkbox: unchecked)
- AC-020: BLCS / PLCSについて、`v1`と`v2`それぞれのCPU smoke testで`dataset load -> model forward -> loss -> metric -> denormalized output -> projection/render`が完走する。 (source checkbox: unchecked)
- AC-021: 共通schema / config、court pose、BLCS physics / gravity / projection / predictor / metric、PLCS generation / loss / predictor / renderer、SLCS scale / uncertaintyのunit・integration testが両versionの契約を固定する。 (source checkbox: unchecked)
- AC-022: README・config comment・dataset / checkpoint schema documentationが`v1` / `v2`の式、default、単位、互換性、mismatch時のエラー、artifact命名・移行方法を単一の正本へ導く。 (source checkbox: unchecked)

The source checkbox state is metadata only, not proof of implementation. The validator must independently verify every item.

## Title

feat: court座標正規化v2を追加し、v1/v2切替と後方互換を実装する

## Body

## 概要

- 現行の軸別court座標正規化を`v1`として維持し、center-to-baseline距離`HALF_LENGTH = 11.885m`をXYZ全軸へ適用する共通scale正規化を`v2`として追加する。
- Hydra config、dataset metadata、checkpoint metadata、推論設定から`v1` / `v2`を明示的に切り替えられるようにし、既存dataset・checkpoint・実験コマンドの後方互換を保つ。
- `v2`では正規化空間で物理コートのアスペクト比を維持し、同じ物理距離の誤差が軸だけを理由に異なるposition lossを受けない座標契約にする。

## 課題

- 現行契約は`src/utils/schema/court.py`で`x_norm=X/5.485`, `y_norm=Y/11.885`, `z_norm=Z/1.07`と定義されている。これはcourt-relativeな無次元化であり、物理距離を等方的に扱う標準化ではない。
- XYだけを見ても、doubles courtのsidelineとbaselineがともに正規化値`±1`になるため、モデルからは物理的に長方形のコートが正方形として表現される。
- normalized Smooth L1の二次領域を物理誤差で書くと、各軸の係数は`1/scale_i^2`になる。現行scaleでは同じ物理誤差に対するZの係数がXの約26.3倍、Yの約123.4倍となる。
- 実データでもnormalized magnitudeは不均衡である。
  - BLCS `data/blcs_broadcast`: mean absolute XYZ=`[0.677, 0.638, 2.907]`、std=`[0.932, 0.829, 3.212]`。
  - PLCS `data/plcs_broadcast`: mean absolute XYZ=`[0.470, 0.473, 0.780]`。Zはpelvis高さ由来の非ゼロoffsetを持つ。
- scaleは生成データ、loss、gravity prior、Hungarian matching、projection、metric、inference、rendererへ暗黙に伝播している。一部は共有tupleを使わず、BLCS physicsやPLCS rendererでコート寸法を直接参照している。
- 既存dataset / checkpointにはnormalization versionやscaleのmetadata検証がなく、単純に共通定数を書き換えると旧normalized値・旧checkpoint出力を`v2`として静かに誤読する。
- `COURT_COORD_SCALE_XYZ`はSLCSも利用しているため、BLCS / PLCSだけを変更すると統合pipeline、SLCS position / uncertainty換算と契約が不一致になる。

## 提案

- versionedな座標正規化contractを単一のresolver / schemaから提供する。

  ```text
  v1 (legacy):
    scale_xyz = (HALF_DOUBLES_WIDTH, HALF_LENGTH, NET_HEIGHT_POST)
              = (5.485, 11.885, 1.07) m

  v2:
    S = HALF_LENGTH = 11.885 m
    scale_xyz = (S, S, S)

  position_norm = position_m / scale_xyz
  position_m = position_norm * scale_xyz
  ```

- Hydraの共通設定に、例えば`court_coordinate_normalization.version: v1 | v2`という必須の選択肢を設ける。初回導入時のdefaultは`v1`とし、既存コマンドの数値挙動を維持する。新baseline・新dataset生成configでは`v2`を明示する。
- 物理コート寸法`HALF_DOUBLES_WIDTH=5.485m`, `HALF_LENGTH=11.885m`, `NET_HEIGHT_POST=1.07m`自体は変更しない。`v2`正規化空間ではdoubles sidelineが`x=±5.485/11.885≈±0.4615`、baselineが`y=±1`、net postが`z≈0.0900`となり、物理アスペクト比を保持する。
- normalized position / translationと、その時間微分として正規化されるvelocityへ、選択されたversionの同じscale contractを注入する。world座標配列の単位`[m]` / `[m/s]`は維持する。
- BLCS / PLCS / SLCSの生成、loader、loss / matching / prior、projection、metric、inference、visualization / analysisが、process-globalな固定scaleではなく同じversioned contractを受け取るようにする。特に以下の暗黙consumerを監査する。
  - `src/tasks/blcs/generate_dataset/simulation/ball_physics.py`
  - `src/tasks/blcs/models/components/differentiable_projection.py`
  - `src/tasks/blcs/training/losses.py`とgravity `height_scale`
  - `src/tasks/plcs/generate_dataset/scene_generator.py`
  - `src/tasks/plcs/data/targets.py`
  - `src/tasks/plcs/visualization/rendering/scene_renderer.py`
  - `src/utils/geometry/court_pose.py`
  - `src/tasks/slcs/{data,evaluation,training,model_io,scripts}`のscale consumer
- `v2`のnormalized Smooth L1の`beta`を監査し、全軸で同じ物理Huber遷移点になることを明文化する。旧scale差の補正として導入されたaxis weightは`v2`configへ持ち込まず、タスク固有の重みを残す場合は正規化補正と区別して根拠を記録する。`v1`configの既存weightと数値挙動は維持する。
- 新しく生成するdatasetとcheckpointにはnormalization version、`scale_xyz`、単位を必ず保存する。runtime config、dataset、checkpoint間のversion / scale不一致はfail-fastで拒否する。
- metadataを持たない既存artifactは、runtimeが`v1`を明示した場合に限りlegacy `v1`として扱う。`v2` runtimeからmetadata欠落artifactを読むこと、versionをshapeや値域から推測すること、mismatchを自動変換することは禁止する。
- 既存dataset / checkpointは上書きせず、`v2` datasetを別versionとして生成し、`v2` baselineを再学習する。旧checkpointのweight自動変換は行わない。

## スコープ

- 対象:
  - `v1` / `v2` scale mapping、typed config / schema、versioned normalize / denormalize API
  - `src/utils/geometry`、`src/tasks/base`の共通変換とconfig伝播
  - BLCS / PLCSのdataset生成・保存・読込、standard / tracking loss、Hungarian matching、physics prior、projection、metric、inference、renderer / analysis
  - scale contractを共有するSLCSのdataset、adapter、evaluation、metric、inference、uncertainty換算
  - dataset / checkpointのnormalization metadata、compatibility guard、設定・README・テスト
  - `v2` datasetの生成と`v2` baselineの再学習・物理m metricによる評価
  - 既存`v1` dataset / checkpoint / configの回帰検証
- 対象外:
  - コート、ネット、camera/world座標そのものの物理寸法・軸向き・原点
  - image / UV / camera intrinsicの正規化
  - SMPLのY-up→court Z-up変換、yaw、canonical poseのroot-relative meter座標
  - `metre`契約で保存されているproduction synthetic PLCS配列の再スケール
  - BLCSのshot samplingやapex上限など、物理軌道生成ポリシーの変更
  - `v1` checkpointを`v2`へ自動変換すること

## Acceptance checklist

- [ ] versioned contractの単一正本が、`v1=(5.485,11.885,1.07)m`、`v2=(11.885,11.885,11.885)m`を返し、未知versionを明示的エラーにする。
- [ ] Hydraの共通configからBLCS / PLCS / SLCSの生成・学習・評価・推論へ同じnormalization versionが伝播し、`v1` / `v2`を明示的に切り替えられる。
- [ ] 初回導入時のdefaultが`v1`であり、既存config・dataset・checkpointを用いた代表的な推論、metric、lossの数値が変更前と許容誤差内で一致する。
- [ ] metadataを持たない既存dataset / checkpointは`v1` runtimeでのみ利用でき、`v2` runtimeでは明示的エラーになる。
- [ ] runtime config、dataset metadata、checkpoint metadataのversionまたは`scale_xyz`が一致しない場合、resume・evaluation・inferenceが明示的エラーになる。
- [ ] 任意shape`(...,3)`の物理positionについて、BLCS / PLCS / SLCSの各versionの`normalize -> denormalize`が最大絶対誤差`1e-5m`以下で元の値を復元する。
- [ ] `v2`正規化空間でdoubles sideline、baseline、net postがそれぞれ`x≈±0.4615`, `y=±1`, `z≈0.0900`となり、物理コート寸法は変更されない。
- [ ] BLCSの`ball_pos_norm`生成、position / velocity decode、differentiable projection、standard / tracking metricが選択versionの同じscale contractを使う。
- [ ] BLCS gravity priorのnormalized second differenceが選択versionのZ scaleに対する`-g*dt^2/scale_z`と一致し、固定値を使うtracking gravity targetもversionごとの整合が検証されている。
- [ ] PLCSのposition生成・target復元、standard / tracking predictor、metric、canonical-to-world統合、3D / top-down rendererが選択versionの同じscale contractを使う。
- [ ] PLCSの`canonical_pose_3d`などroot-relative meter座標は再スケールされず、position translationだけがversioned normalized contractを使う。
- [ ] SLCSのposition、evaluation、metric、adapter、inference、uncertaintyのnormalized↔meter変換が選択versionへ追随し、統合`SceneResult`は引き続きcourt/world`[m]`を返す。
- [ ] `v2`で同じ物理position誤差をX/Y/Zへ個別に与えたとき、defaultのunweighted Smooth L1 lossが全軸で一致し、共通の物理Huber遷移点が設定・テスト・ドキュメントで確認できる。
- [ ] `v2`のBLCS / PLCS default position lossとHungarian position costに、旧axis-scale補正由来の非等方weightが残っていない。`v1`の既存config挙動は維持される。
- [ ] 新規生成dataset metadataにnormalization version、`scale_xyz`、position / velocity単位が保存され、root / scene間のmissing・unknown・mixed contractをloaderが拒否する。
- [ ] 新規checkpoint metadataにnormalization versionと`scale_xyz`が保存され、checkpoint由来のversionが推論時に復元・検証される。
- [ ] 既存`v1` dataset / checkpointは上書きされず、`v2` dataset / checkpointとartifact名・metadataの両方で識別できる。
- [ ] BLCS / PLCSの`v2` datasetが別versionとして生成され、保存normalized値をmetersへ戻した値が生成時world値と最大絶対誤差`1e-5m`以下で一致する。
- [ ] `v2`のBLCS / PLCS baselineを再学習し、`v1` baselineとの比較を物理m単位の軸別metricと統合metricで記録する。
- [ ] BLCS / PLCSについて、`v1`と`v2`それぞれのCPU smoke testで`dataset load -> model forward -> loss -> metric -> denormalized output -> projection/render`が完走する。
- [ ] 共通schema / config、court pose、BLCS physics / gravity / projection / predictor / metric、PLCS generation / loss / predictor / renderer、SLCS scale / uncertaintyのunit・integration testが両versionの契約を固定する。
- [ ] README・config comment・dataset / checkpoint schema documentationが`v1` / `v2`の式、default、単位、互換性、mismatch時のエラー、artifact命名・移行方法を単一の正本へ導く。

## 補足

- 現在の`data/blcs_broadcast`と`data/plcs_broadcast`は2026-07-05生成でnormalization metadataを持たないため、明示的なlegacy `v1` artifactとして扱う。
- BLCS旧データの28m級高弾道は物理生成データの問題であり、`v2`へ変更しても物理高度は変わらない。`v2` dataset生成時に分布を記録するが、apex policy自体は本Issueの対象外とする。
- 既存knowledgeの`position_axis_weights=[1,4,1]`やtracking`[1,1,0.5]`は`v1`前提である。`v1`再現では維持し、`v2` baselineへは根拠なく持ち込まない。
- Reference: #719
- Reference: #695
- Reference: #779
- Reference: #169

