# Tests

- Issue: #786
- Attempt: 1
- Test cycle: 1
- Status: PENDING
- Frozen acceptance checklist SHA-256: `95bcebf4388fdba9773e3c538c9e22caf82b6e4a413ec1241e9a58b0c4483032`
- Candidate SHA-256: `PENDING`

## Candidate identity

## Acceptance-checklist-to-test mapping

| ID | Issue checklist item | Test or authoritative evidence | Result |
|---|---|---|---|
| AC-001 | versioned contractの単一正本が、`v1=(5.485,11.885,1.07)m`、`v2=(11.885,11.885,11.885)m`を返し、未知versionを明示的エラーにする。 | PENDING | PENDING |
| AC-002 | Hydraの共通configからBLCS / PLCS / SLCSの生成・学習・評価・推論へ同じnormalization versionが伝播し、`v1` / `v2`を明示的に切り替えられる。 | PENDING | PENDING |
| AC-003 | 初回導入時のdefaultが`v1`であり、既存config・dataset・checkpointを用いた代表的な推論、metric、lossの数値が変更前と許容誤差内で一致する。 | PENDING | PENDING |
| AC-004 | metadataを持たない既存dataset / checkpointは`v1` runtimeでのみ利用でき、`v2` runtimeでは明示的エラーになる。 | PENDING | PENDING |
| AC-005 | runtime config、dataset metadata、checkpoint metadataのversionまたは`scale_xyz`が一致しない場合、resume・evaluation・inferenceが明示的エラーになる。 | PENDING | PENDING |
| AC-006 | 任意shape`(...,3)`の物理positionについて、BLCS / PLCS / SLCSの各versionの`normalize -> denormalize`が最大絶対誤差`1e-5m`以下で元の値を復元する。 | PENDING | PENDING |
| AC-007 | `v2`正規化空間でdoubles sideline、baseline、net postがそれぞれ`x≈±0.4615`, `y=±1`, `z≈0.0900`となり、物理コート寸法は変更されない。 | PENDING | PENDING |
| AC-008 | BLCSの`ball_pos_norm`生成、position / velocity decode、differentiable projection、standard / tracking metricが選択versionの同じscale contractを使う。 | PENDING | PENDING |
| AC-009 | BLCS gravity priorのnormalized second differenceが選択versionのZ scaleに対する`-g*dt^2/scale_z`と一致し、固定値を使うtracking gravity targetもversionごとの整合が検証されている。 | PENDING | PENDING |
| AC-010 | PLCSのposition生成・target復元、standard / tracking predictor、metric、canonical-to-world統合、3D / top-down rendererが選択versionの同じscale contractを使う。 | PENDING | PENDING |
| AC-011 | PLCSの`canonical_pose_3d`などroot-relative meter座標は再スケールされず、position translationだけがversioned normalized contractを使う。 | PENDING | PENDING |
| AC-012 | SLCSのposition、evaluation、metric、adapter、inference、uncertaintyのnormalized↔meter変換が選択versionへ追随し、統合`SceneResult`は引き続きcourt/world`[m]`を返す。 | PENDING | PENDING |
| AC-013 | `v2`で同じ物理position誤差をX/Y/Zへ個別に与えたとき、defaultのunweighted Smooth L1 lossが全軸で一致し、共通の物理Huber遷移点が設定・テスト・ドキュメントで確認できる。 | PENDING | PENDING |
| AC-014 | `v2`のBLCS / PLCS default position lossとHungarian position costに、旧axis-scale補正由来の非等方weightが残っていない。`v1`の既存config挙動は維持される。 | PENDING | PENDING |
| AC-015 | 新規生成dataset metadataにnormalization version、`scale_xyz`、position / velocity単位が保存され、root / scene間のmissing・unknown・mixed contractをloaderが拒否する。 | PENDING | PENDING |
| AC-016 | 新規checkpoint metadataにnormalization versionと`scale_xyz`が保存され、checkpoint由来のversionが推論時に復元・検証される。 | PENDING | PENDING |
| AC-017 | 既存`v1` dataset / checkpointは上書きされず、`v2` dataset / checkpointとartifact名・metadataの両方で識別できる。 | PENDING | PENDING |
| AC-018 | BLCS / PLCSの`v2` datasetが別versionとして生成され、保存normalized値をmetersへ戻した値が生成時world値と最大絶対誤差`1e-5m`以下で一致する。 | PENDING | PENDING |
| AC-019 | `v2`のBLCS / PLCS baselineを再学習し、`v1` baselineとの比較を物理m単位の軸別metricと統合metricで記録する。 | PENDING | PENDING |
| AC-020 | BLCS / PLCSについて、`v1`と`v2`それぞれのCPU smoke testで`dataset load -> model forward -> loss -> metric -> denormalized output -> projection/render`が完走する。 | PENDING | PENDING |
| AC-021 | 共通schema / config、court pose、BLCS physics / gravity / projection / predictor / metric、PLCS generation / loss / predictor / renderer、SLCS scale / uncertaintyのunit・integration testが両versionの契約を固定する。 | PENDING | PENDING |
| AC-022 | README・config comment・dataset / checkpoint schema documentationが`v1` / `v2`の式、default、単位、互換性、mismatch時のエラー、artifact命名・移行方法を単一の正本へ導く。 | PENDING | PENDING |

## Tests added or changed

None

## Normal, boundary, invalid, and regression cases

## Canonical command results

## Commands and exact outcomes

## Failures encountered

None

## Untested risks and reasons

None

## Final test verdict

PENDING

## RETURN implementation findings

None
