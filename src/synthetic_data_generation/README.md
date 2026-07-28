# `src/synthetic_data_generation`

3D Gaussian scene をコート座標へ対応付け、movable Gaussian asset を
RGB overlay なしで scene に合成する境界です。外部 COLMAP/3DGS artifact
の immutable export、court alignment、受理済み `SceneContract`、および
versioned composition contract を担当します。

旧 `dataset/`、`provider/`、`rendering/` は撤去しました。2D overlay と
B00 固有 publication を前提にした実装は、新しい 3DGS-native な
BLCS/PLCS/court generation へ流用しません。

実装全体の結果、制約、metrics、GIF/contact sheet は
[3DGS-native synthetic data 最終レポート](../../docs/3dgs-native-synthetic-data/README.md)
にまとめています。

## Modules

### `scene_contract.py`

undistort/crop 後の OpenCV camera、content-addressed artifact、受理済み
`T_scene_from_court` を保持する versioned JSON 契約です。未知 version、
反射、不整合 inverse を拒否します。

### `alignment/scene_provider/`

read-only の COLMAP/3DGS artifact を camera、point cloud、RGB、source hash
を持つ検証済み bundle へ export します。外部 repository の application
module や directory layout は alignment domain へ公開しません。

### `alignment/artifacts/` と `alignment/components/`

fit/holdout を先に分離し、court detector と ground-line evidence から
metric court template を当てます。calibration、holdout validation、明示的
な user override を immutable artifact として段階的に公開します。

### `composition/`

背景と movable asset の Gaussian tensor、metric-local 座標、Sim(3)
placement、instance identity、artifact provenance を扱います。means、
quaternion、anisotropic scale を同じ変換で更新し、すべてを concat して
から単一 renderer call へ渡します。

NHT feature は各 deferred shader 固有の latent appearance です。そのため
独立に学習された asset を暗黙に concat せず、background と全 asset の
`appearance_space_sha256` が一致する場合だけ合成します。後続の ball /
human asset 構築は共有または frozen appearance space を明示的に作る必要が
あります。

### `blcs/`

既存 `src/tasks/blcs/generate_dataset/` の物理 trajectory を入力にし、
Gaussian ball の persistent instance placement とラベルへ変換します。

- `assets.py` はユーザー提供 ball asset の明示 registry を扱います。全
  tensor / appearance / provenance の local byte を SHA-256 と size で検証し、
  seed・scene/object key・registry fingerprint から variant を決定します。
  欠損時の既定 asset や独立 appearance space への fallback はありません。
- `asset_ingestion.py` は source asset を metre 単位・ball-centred の NHT
  tensor pack へ正規化し、直径と原点を数値検証して registry を atomic に
  公開します。背景と同じ frozen appearance を使う native NHT source、または
  `passed` conversion report、target shader/tensor hash、20 dB 以上の validation
  PSNR を持つ frozen-target optimization 結果だけを受理します。vanilla SH
  PLY と独立 NHT feature を宣言だけで通す経路はありません。
- `planner.py` は single `[T,3]` と multi `[T,O,3]` を同じ global timeline
  へ正規化し、`num_balls` より後の padding column を除外します。object
  column と positive `instance_id` の対応は birth/death をまたいで固定です。
- court-space metre trajectory に受理済み `T_scene_from_court` を適用し、
  各 frame の `T_scene_from_asset`、全 camera の OpenCV pixel/depth、および
  geometric in-frame visibility を immutable label plan に保存します。

`camera_geometric_visible` は occlusion を表しません。また、unit test の
synthetic asset は registry mechanics の検証専用であり、ユーザー提供 ball
asset の受理を意味しません。

Plan の strict load 後は、隔離環境の
`third_party/nht/blcs_render.py` が active instance だけを背景 Gaussian と
concat し、frame ごとに一体レンダリングします。RGB/alpha/expected-depth と
完全な plan label を同じ immutable output に束ねます。同じ Gaussian scene
を one-hot instance 色で追加 rasterize し、alpha-composited contribution
AOV、per-instance mask、exclusive segmentation を生成します。NHT feature
へ instance channel は追加しないため、deferred appearance は変化しません。
旧 `projected-centre-depth-consistency-v1` は exact AOV との比較値として
のみ残し、2回の public renderer call と NHT 内部 depth auxiliary pass を
manifest に明記します。

### `plcs/`

`planner.py` は single/multi-person の identity、court-metre footprint、
velocity、yaw、SMPL-X pose index、presence を seed から決定的に生成します。
instance ID は全 timeline で固定され、欠落 frame、court 外 footprint、
multi-person collision、未知 pose を拒否します。受理済み
`T_scene_from_court` と人物 asset の軸変換は
`src/synthetic_data_generation/plcs_avatar/prototype_plcs_plan.py` が明示的な Sim(3) にし、
export 済み camera での投影とともに immutable plan へ保存します。

隔離 NHT runtime の `plcs_render.py` は pose ごとの 4,096 Gaussian asset を
背景と concat してから一体レンダリングし、RGB/alpha/depth に加えて exact
instance contribution、mask、segmentation、bbox、identity、pose、3D placement
を公開します。`p5_acceptance.py` が same-seed byte identity、時間整合、
投影誤差、可視 pixel 数、AOV alpha 整合を threshold 付きで判定します。

### `court/`

`layout.py` は受理済み alignment の reference court と、同じ検証済み
geometry artifact に含まれる追加 court candidate を N 個の物理 instance
として読み込みます。B00 では `court_0` / `court_1` の二面を保持し、推測した
offset court への fallback はありません。

`labels.py` は各面の14 line keypointを near/far 対称な7 semantic classへ
圧縮します。annotation は `court_instance_id` と14物理点の
UV/depth/in-frame/renderer visibilityを保持しますが、model targetは全
instanceを同一classへpixelwise-maximumで統合した7-channel multi-peak
heatmapです。二面では1 channelに最大4 peakが入ります。instance割当、
homography、Hungarian matchingはpost-processであり学習targetではありません。

`novel_view.py` は0.25 m / 1.5 degreeの保守的SfM局所baselineを提供します。
`orbits.py` は実測SfM envelopeから、二面complexまたは各courtを内向きに見る
circle/ellipse、内外scale、複数heightの滑らかなfamilyを生成します。全14点を
要求せず、full/near-full/partial/sparse coverageを明示的に保持します。
`release.py` は隣接frame漏洩を防ぐためfamily全体を一つのsplitへ固定し、
validation/testの双方でshape/scale/targetをcoverします。要求候補数や
semantic coverageを満たせない場合にgateを緩和するfallbackはありません。

## Export-first alignment pipeline

検証は必ず scene-provider export から開始します。

```bash
.venv/bin/python \
  -m src.synthetic_data_generation.scripts.alignment.export_scene_provider

.venv/bin/python \
  -m src.synthetic_data_generation.scripts.run_alignment_pipeline \
  jobs=b00 \
  stages=all
```

各 alignment stage は同じ `run(cfg)` 実装を使って単独実行できます。

```bash
.venv/bin/python \
  -m src.synthetic_data_generation.scripts.alignment.infer_ground_line_map
.venv/bin/python \
  -m src.synthetic_data_generation.scripts.alignment.fit_ground_courts
.venv/bin/python \
  -m src.synthetic_data_generation.scripts.alignment.calibrate_court_alignment
.venv/bin/python \
  -m src.synthetic_data_generation.scripts.alignment.finalize_court_alignment
```

`resume=true` では artifact の schema、fingerprint、SHA-256、provider
fingerprint、前段 artifact identity が一致する場合だけ再利用します。
fit calibration が失敗した場合は finalization を実行せず、holdout rejection
から `SceneContract` を作る場合は設定済み user override declaration との
完全一致を要求します。
