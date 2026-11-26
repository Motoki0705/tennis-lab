# #2605 の実装指示（親 #1722）

Issue #2605 の本文が仕様の正本です。人間が設計資料（https://claude.ai/artifact/BovH3XfenxBZJZ8U1qWAXL）の8項目の推奨を「推奨でよい」と承認し、親の合意（改訂3）に反映済みです。本文の範囲と受入条件のとおりに実装し、PR を作ってください（本文は日本語、`Closes #2605`）。

## 親が調べた事実（照合して使う）

- SceneResult の正本は `src/tennis_scene/schema.py`。`schema_version` は今 1 と 2 だけを受け付け、`validate_scene_result_arrays` が mask・理由コード・0埋めを検証している。v3 はこの規則を保ったまま、イベント項目だけを足す。
- パイプラインの構成は `src/tennis_scene/pipeline/contracts.py` の `STANDARD_COMPONENTS`（20個、最後が `scene_assembly`）と `src/tennis_scene/pipeline/README.md`。`exports.scene` は今 scene_assembly を入力にしている。外部注釈を method として取り込む前例は `ball_detection.method=gpt_annotation`。
- 既存のイベント検出: 規則は `src/tennis_scene/ball_events/`（`detect_scene_events(scene)`、出力 `ball_event_predictions.v1`、#2568/PR #2569）。モデルは `ball_refiner_3d.inference.scene.predict_scene_events()`（#2567/PR #2571、checkpoint 契約 `ball_refiner_3d.events.v3`、`fps_tolerance` あり）。クラス順は shot・bounce・net。
- multiview-v2 は `data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v2/`。各 clip の `scene/scene.json`（`tennis_scene_index_v1`）が components 6個（multiview_scene_inputs, body_view_selection, hmr2_features, gvhmr, body_placement, scene_assembly）と export を指す。v2 を作った `src.tennis_scene.multiview_scene` は main になく（#1739 の draft PR #2470）、これに依存しない。v2 は v1 から hard link で派生し、manifest の `base` と `provenance` に来歴を残している。v3 も同じ形で作る。
- 注釈: `data/tennis_multiview/processed/meiji_4cam/dataset/videos/video_000/clip_000-068_ball_events/`。
  - 69ファイル（`clip_NNN_ball_events.json`、schema `meiji_event_labels.v1`）。親の集計では全913件（shot 421・bounce 457・net 35、net は32クリップ）、status は certain 809・probable 91・uncertain 13、判定不能は33区間・計1,885 frame。
  - fps・frames は v2 manifest と一致する。
  - `player_hint` の基準カメラは、65クリップの notes で cam0 と書かれている。clip_026・034・036・039 は未記載。
  - `labeler` は13通りの表記がある。`role` キーは clip_040・043 にだけある。
  - 同じフォルダの `meiji_event_labels.v1.schema.json`、`clip_027_ball_events.example.json`、`visualizations/`、`*:Zone.Identifier` は取り込まない。
  - このフォルダはユーザーの成果物なので、変更・移動・削除しない。
- #2542 の採点用6クリップ（test に入れる）は clip_027・028・040・043・056・064。調整用は clip_001・009（記録は PR #2597）。

## 進め方の注意

- 実装前に、v3 の配列・metadata・成果物 JSON の形と、event_detection の設定・recipe への入れ方を短く設計し、PR 本文に書く。本文の案から変える場合は理由を書く。
- スキーマは `src/tennis_scene` の下流全体に効く。`schema_version` を見ている箇所を repo 全体で洗い出し、v3 対応とテストを揃える。テストは tests/README.md の方針で、拒否条件ごとに書く。
- 静かなフォールバックを作らない。method 未指定、注釈と scene の不一致、出どころと項目の不一致は止める。
- コードの fingerprint は src 全体にかかる。v3 の作成は、マージ予定の HEAD のコードで最後にやり直し、v2 の配列との bit 一致と件数を照合する。生成物は data 配下（git 外）に置き、作成コマンドと照合結果を PR に書く。
- 基準未記載の4クリップは、重ね合わせ動画と cam0 映像で手前・奥を確かめ、確認した frame と結果を PR に記録する。cam0 と矛盾する場合は変換せずに止めて報告する。
- GPU は使わない想定。使う必要が出たら training queue を経由する。
- 範囲外: ポストトレーニングとアブレーション（#2600）、パイプライン全体の再実行、予測を v3 に入れること。
