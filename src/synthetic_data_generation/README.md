# `src/synthetic_data_generation`

再構成済み3Dテニスシーンへ物理シミュレーション由来のボールを合成し、学習用データセットを公開するパイプラインです。`src/tennis_scene` の統合推論・可視化とは独立し、外部scene providerの取り込みからcourt alignment、合成レンダリング、dataset publicationまでを担当します。

## Modules

### `scene_contract.py`

外部3D scene providerをrendererやBLCSから分離するversioned JSON契約です。undistort/crop後のOpenCV camera、content-addressed artifact、受理済み`T_scene_from_court`を保持し、未知version・反射・不整合inverseを拒否します。

### `provider/`

read-onlyの3DGS/COLMAP artifactを、camera・point cloud・RGB・source hashを持つ検証済みbundleへ変換します。外部repositoryのapplication moduleやdirectory layoutはdomain codeへ公開しません。

### `alignment/`

fit/holdoutを先に分離し、court detectorとground-line evidenceからmetric court templateを当てます。calibration、holdout validation、明示的なuser overrideをimmutable artifactとして段階的に公開します。

### `rendering/`

captured camera、static RGB/depth、scene-space sphereを扱うtyped portとadapterです。deterministic CPU reference rendererと、3DGSをfile/subprocess境界越しに呼ぶadapterを提供します。

### `dataset/`

BLCS trajectoryを受理済み`SceneContract`でscene座標へ変換し、TrackNet互換の単一フレームpilotおよび複数trajectory・複数camera groupのtraining-only datasetをatomic publishします。

### `scripts/` / `configs/`

provider export、alignment、static-scene smoke、pilot、full-scale dataset publicationのHydra entry pointと設定です。代表的な実行例:

```bash
.venv/bin/python -m src.synthetic_data_generation.scripts.export_scene_provider
.venv/bin/python -m src.synthetic_data_generation.scripts.publish_b00_full_scale_dataset
```

各entry pointの入力artifactとhashは対応する`configs/`で明示し、暗黙のfallbackは行いません。

## Alignment pipeline

Provider bundleをexportした後、4つのalignment stageを統合CLIから直列実行します。

```bash
.venv/bin/python -m src.synthetic_data_generation.scripts.export_scene_provider
.venv/bin/python -m src.synthetic_data_generation.scripts.run_alignment_pipeline \
  jobs=b00 \
  stages=all
```

各stageは同じ`run(cfg)`実装を使って単独実行できます。

```bash
.venv/bin/python -m src.synthetic_data_generation.scripts.alignment.infer_ground_line_map
.venv/bin/python -m src.synthetic_data_generation.scripts.alignment.fit_ground_courts
.venv/bin/python -m src.synthetic_data_generation.scripts.alignment.calibrate_court_alignment
.venv/bin/python -m src.synthetic_data_generation.scripts.alignment.finalize_court_alignment
```

`resume=true`ではartifactのschema、fingerprint、SHA-256、provider fingerprint、前段artifact identityが一致する場合だけ再利用します。fit calibrationが失敗した場合はfinalizationを実行せず、holdout rejectionからSceneContractを作る場合は設定済みのuser override declarationとの完全一致を要求します。
