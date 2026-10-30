---
id: run-court-scene-pair-manifests-v1
type: run
task: synthetic_data_generation
sequence: 29
recorded_at: '2026-10-31'
title: 既存Court poolから新5ペアと4シーン共通評価のmanifestを作成する
issue: 1583
provider: codex
date: '2026-10-31'
status: done
config:
  source_pool: court_scene_diversity_6000_v1
  selection_seed: 42
  pairs: [P02, P03, P12, P13, P23]
  train_per_scene: 2750
  validation_per_scene: 250
  evaluation: all accepted images on the original four-scene validation trajectories
  rerender: false
metrics:
  accepted_source_samples: 19934
  new_manifests: 11
  pair_training_samples: 5500
  pair_validation_samples: 500
  common_evaluation_samples: 2439
  training_conditions_checked: 9
  evaluation_train_sample_overlap: 0
  evaluation_train_trajectory_overlap: 0
  evaluation_train_trajectory_group_overlap: 0
  uploaded_manifests: 11
  changed_preexisting_drive_entries: 0
repro:
  branch: codex/scene-pair-manifests
  command: >-
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. /usr/bin/time -v .venv/bin/python
    knowledge/runs/run-court-scene-pair-manifests-v1/manifests.py create
    --metadata-root tmp/court-scene-pairs-v1/metadata
    --manifest-root tmp/court-scene-pairs-v1/manifests
    --report knowledge/runs/run-court-scene-pair-manifests-v1/manifest-creation.json
    2> tmp/court-scene-pairs-v1/receipts/create-time.log
artifacts:
  run_dir: knowledge/runs/run-court-scene-pair-manifests-v1
  p01_equivalence: knowledge/runs/run-court-scene-pair-manifests-v1/p01-equivalence.json
  verification: knowledge/runs/run-court-scene-pair-manifests-v1/manifest-verification.json
  metadata_download: knowledge/runs/run-court-scene-pair-manifests-v1/metadata-download.json
  consumer_compatibility: knowledge/runs/run-court-scene-pair-manifests-v1/consumer-compatibility.json
  tests: knowledge/runs/run-court-scene-pair-manifests-v1/test-results.json
  cpu_resources: knowledge/runs/run-court-scene-pair-manifests-v1/cpu-resources.json
  drive_publication: knowledge/runs/run-court-scene-pair-manifests-v1/drive-publication.json
  publication_read_recovery: knowledge/runs/run-court-scene-pair-manifests-v1/publication-read-recovery.json
  reader_base_verification: knowledge/runs/run-court-scene-pair-manifests-v1/manifest-verification-on-reader-base.json
  reader_base_compatibility: knowledge/runs/run-court-scene-pair-manifests-v1/consumer-on-reader-base.json
parents:
- run-court-scene-diversity-6000-v1
relations: []
papers: []
tags:
- scene-diversity
- pairwise-comparison
- metadata-only
---

既存の固定6000枚poolから、新5ペアの学習・validation入力と4シーン共通評価の
sample列をCPUで作成した。元の`nested_order`関数をそのまま使い、先に再現した
B00+B01のtrain 5,500枚・val 500枚は、既存scene-2と順序を含めて完全一致した。
さらに既存8 manifestを照合し、4シーンの順位が従来のprefixと一致することを確認した。

## 入力の同一性と評価の分離

新5ペアはいずれも各sceneのtrain先頭2,750枚、val先頭250枚である。
同じsceneの画像はペアをまたいで共通となり、P01の既存学習結果を再利用するための
入力同一性が成立した。選択順の正本は前提runの`generate.py`であり、
本runのscriptに順位計算を複製していない。描画依存をimportせず、同関数のASTだけを実行した。

共通評価は元のval軌道で採択された全画像を含み、B00 1,056・B01 643・B02 462・
B03 278の2,439枚となった。全9 trainに対し、sample、trajectory、学習consumerが
使うtrajectory groupの重複がすべて0だった。各ペアのtrain/val間でも同じ3種類の漏れが0だった。
これらは[実データ照合JSON](../../runs/run-court-scene-pair-manifests-v1/manifest-verification.json)に記録した。
従来の学習時の合成val 500枚と、今回の共通評価全体は異なる母集団として扱う。

新11 manifestを元poolと同じDriveディレクトリへ追加し、3,182,035 bytesの内容一致を確認した。
[公開receipt](../../runs/run-court-scene-pair-manifests-v1/drive-publication.json)はDrive IDを保持し、
既存2,422ファイルを含む3,790 entryのID・型・サイズ・内容・更新日時が不変だったことを記録する。
公開前のinventoryは一度失敗し、初版wrapperが詳細JSONを失ったため原因は未確定である。
[回復記録](../../runs/run-court-scene-pair-manifests-v1/publication-read-recovery.json)のとおり、
独立したinventoryの再取得と前後比較が成功してから公開した。失敗理由の保持を修正し、回帰テストを追加した。

## 検証範囲と制約

Driveから必要なメタデータ919ファイル、37,928,809 bytesだけを取得した。
descriptor/index・accepted membership・固定済み軌道割当を検証し、store単位で
疎recordを読む。JPEGを取得せず、既存recordやsplitを書き換えていない。
実consumerの`read_subset`でも、新11 manifestのschema・source参照・配置を確認した。
作成・照合のCPU最大RSSは約485MiBだった。

[テスト記録](../../runs/run-court-scene-pair-manifests-v1/test-results.json)は、
基底PR #1594上でbundle固有15件と既存consumerの9件が計61秒で成功した記録を保持する。
新規caseは最大0.203秒で、既存のpublication fixtureを使うcaseは4.634〜10.952秒だった。
全画像の読込・教師やQAの再検証、学習・モデル評価は実施していない。
データ選択処理のためTensorBoard曲線は該当しない。

作成開始時のmainでは、実poolのmethod `saved_train_trajectories`を`TrajectoryMethod`が
拒否することも[CPUで確認](../../runs/run-court-scene-pair-manifests-v1/consumer-compatibility.json)した。
これはmanifestのsample列や配置とは独立した、PR #1570で取り除いた旧publication readerの問題である。
親#1581へ報告し、本runからvalidatorの迂回や担当外の復活は行っていない。
その修復PR #1594の指定headへ積み直し、enumが復元されたことと、全manifestの再照合成功を確認した。
全publicationの厳密な読込は、旧学習コード・修復版の両方でラップトップ上の幾何の
末尾桁差が観測されているため、親が前回と同じColab環境で確認する方針である。
その確認後に、同じ条件の学習・共通評価を実行する。
今回の入力検証から、シーン組合せによるモデル精度や汎化の効果は判断しない。

ファイル名・Drive配置・DATA root相対の学習設定と再現手順は
[bundle README](../../runs/run-court-scene-pair-manifests-v1/README.md)を参照する。
