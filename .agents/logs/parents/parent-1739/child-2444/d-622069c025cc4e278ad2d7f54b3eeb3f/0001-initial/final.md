#2444 の初期案は、**#2241 のボール由来の動画共有判定を入力にし、関節レイで絞った骨格候補をフレーム単位の MILP で割り当てる**構成です。親に確認してほしい判断は、court side の入力方式と、MILP を初期ソルバーにするかの2点です。

## 入力と court side

#2363 の実ストアで、停止した14 clipを含む **69 clip・276 camera/clip・1,332,834検出行**を確認しました。`person_detections` v2 と `person_pose_features` v2 は、全行で `frame_offsets` と `source_rows` が一致しました。`local_court_calibration` v1 も全件にあります。新経路はこの3成果物と向きの判定を入力にし、旧追跡・選別・対応付けや、それらに依存する `aligned_cameras` が入力の依存関係にあれば拒否する設計です。 [実ストアの照合結果](https://github.com/Motoki0705/tennis-lab/blob/3dbdefdd5e656ffc915c30ac26d7ae06efd1e22f/knowledge/runs/run-i2444-multiview-pose-probe/README.md#meiji_4camの入力と向き)

| 選択肢 | 判断材料 |
|---|---|
| **(a) ボールで判定** | ボールは1個なので人物の割当が不要です。#2363 はボール無効。#2241 の実測は不足した65 clipの検出・点生成・校正補完に67分39秒で、同条件の全69 clipは約70分の見積もりです。 [#2241 の計測](https://github.com/Motoki0705/tennis-lab/blob/8d63f4382fb2ff38dd701bbdd99677b971a1d57f/knowledge/nodes/court_side/000007-run-court-side-video-shared-20261118.md) |
| **(b) ボール由来の動画共有判定を入力にする（推奨）** | #2241 の保存済み判定と #2363 は、全69 clipで元動画の記録SHA・カメラ行列・共有コートとdriftが一致しました。追加GPU推論なしで使えます。既存の共有sideコンポーネントはclipごとのボール点も要求するため、新経路では検証済み判定をdigestで固定して読む入口が必要です。 [共有判定の設定と再現](https://github.com/Motoki0705/tennis-lab/blob/8d63f4382fb2ff38dd701bbdd99677b971a1d57f/knowledge/runs/run-court-side-video-shared-20261118/reproduce.md) |
| **(c) 人物レイで判定** | 肩・腰のレイで8仮説を採点したCPU試作では、旧設定の確定 **68/68**、ボール共有判定 **69/69** と一致しました。旧判定が停止した `clip_012` でも同じ向きが首位でした。3,887 frameの再実行は入力照合込みで29.68秒。共有判定がない動画で使う候補で、別収録での検証が残ります。 [全仮説の結果](https://github.com/Motoki0705/tennis-lab/blob/3dbdefdd5e656ffc915c30ac26d7ae06efd1e22f/knowledge/runs/run-i2444-multiview-pose-probe/side.json) |

## 骨格候補と割当

**候補を絞ってから MILP で選ぶ案**を推奨します。4D Association Graph の視点間レイ整合と観測の排他制約を参考に、同じフレームの検出行を頂点にします。カメラ対ごとの行ペアを比較し、整合するペアを多視点候補へ広げます。候補ごとに頑健な三角測量を行い、関節ごとの再投影残差、視点間の支持、骨長から混入を検査します。MILPでは各行の採用を高々1回とし、割当なしも許します。ダブルスでも人数を固定せず、競合が接戦の候補は理由を残します。 [#1737 の調査](https://github.com/Motoki0705/tennis-lab/pull/1737)・[4D Association Graph §4–5](https://arxiv.org/html/2002.12625v1#S4)

- **全組合せを MILP に渡す案**は、小さい合成例の比較基準に使えます。候補数が視点ごとの検出行数の積で増えるため、実データでは幾何で絞り、観測を取り合う候補群ごとに解く方が扱いやすいです。
- **貪欲法**は、重み順に未使用行を確保すれば排他性を保てます。早い選択が後続の良い組合せを塞ぐため、同じ候補を使う速度比較に適しています。
- **Meiji 3cam の実測**では、人手ラベル付き4 clip・3,497 frameの micro pair F1 は貪欲法 **0.8472**、MILP **0.8485**。選手同士の組では **0.8676 → 0.8728** でした。一方、`video_001/clip_001` は **0.8802 → 0.8640**。ソルバー単体の clip 別P95は貪欲法 **0.010–0.015 ms**、MILP **7.96–8.68 ms** でした。肩・腰の試作では手首の混入を検出できなかったため、全身の検査とダブルスでの精度を採用判断に使います。 [clip別の採点結果](https://github.com/Motoki0705/tennis-lab/blob/3dbdefdd5e656ffc915c30ac26d7ae06efd1e22f/knowledge/runs/run-i2444-multiview-pose-probe/pair-f1.json)

## Dへの受け渡しと検証

Dには、元フレーム番号と `frame_offsets`、フレーム内で有効な候補ID、コート座標のCOCO17、カメラごとの `source_row`、関節ごとの支持視点・再投影残差・有効maskを渡す案です。校正とsideの版を保存し、未割当の行には視点不足、退化、混入、競合などの理由を残します。

実装は **入力契約と向き → 骨格候補と排他割当 → 実データ評価** の順がよいです。単体・結合テストでは旧成果物の混入、行の再使用、別人の腕が入った合成例を検査します。実データでは Meiji 3cam の4 clipと #2442 の注釈済みframeを採点し、疑義frameと無作為標本のF1・被覆を併記します。未使用視点の検査は、その視点を候補生成・割当・三角測量から外して再実行し、同一人物の関節への再投影誤差と採点可能率を測ります。

試作と記録は `codex/parent-1739-issue-2444-primary` の `3dbdefdd5e656ffc915c30ac26d7ae06efd1e22f` に push 済みです。既存テスト62件、ruff、mypy、knowledge検査が通り、再実行では時間欄以外が一致しました。 [#2444 の検証記録](https://github.com/Motoki0705/tennis-lab/issues/2444#issuecomment-7107721305)

```bash
bash knowledge/runs/run-i2444-multiview-pose-probe/reproduce.sh
```

PARENT_RESULT {"schema_version":1,"state":"needs_parent","wait":null}