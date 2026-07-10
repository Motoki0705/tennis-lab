# `src/tennis_scene/clip_studio`

長時間・非同期のマルチカメラ試合動画を共有グローバルタイムライン上で同期し、ラリー単位のクリップを切り出して `src/tennis_scene` パイプラインの入力契約（全カメラで fps / フレーム数 / 解像度が一致）に合う形でエクスポートする GUI ツールです。

## 同期の規約

全モジュールで `local_time = global_time + offset_sec` を使う。ソースはグローバル区間 `[-offset, duration - offset]` をカバーし、クリップはグローバル時刻の半開区間 `[start_sec, end_sec)`。

## Modules

- **`project.py`**: `ClipSource` / `Clip` / `ClipStudioProject`。プロジェクトの JSON 永続化（atomic write、相対パスはプロジェクトファイル基準で解決）。
- **`timeline.py`**: グローバル時刻↔ソースフレームの写像（`source_frame_index`）、カバレッジ、`TimelineGeometry`（px↔sec、ヒットテスト）。
- **`state.py`**: `ClipStudioState`。再生・シーク、オフセット調整、in/out マーク、クリップ作成/削除/選択、タイムラインのズーム/パン。cv2 非依存の純ロジック。
- **`sources.py`**: `PreviewSource`（smart-seek + タイル幅への縮小 + LRU キャッシュ）と `PreviewSourcePool`（カメラ並列フェッチ）。長尺動画でもプレビューが軽い理由はここ。
- **`render.py`**: `render_studio` / `compute_layout`。タイル格子 + タイムライン（ルーラー、クリップ行、カメラ別カバレッジ行、マーク、プレイヘッド）+ フッターの純描画。
- **`imaging.py`**: `letterbox_frame` / `LetterboxSpec`。レンダラとエクスポータで共有するアスペクト保存フィット。
- **`audio_sync.py`**: 音声エンベロープの FFT 相互相関による `offset_sec` 自動推定（分解能は envelope_rate 依存、既定 10ms）。
- **`export.py`**: `plan_clip_export`（純粋・検証込みのフレーム写像）と `export_clip(s)`（ストリーミング再エンコード + `clip.json` マニフェスト + 書き出し後の contract 自己検証）。
- **`app.py`**: `ClipStudioApp`。cv2 ウィンドウ・キー/マウスイベント・再生クロック。差分レンダリングでアイドル時は再描画しない。

## 使い方

```bash
# 新規プロジェクト（recording_id は後日の追記でも変わらない収録ID）
.venv/bin/python -m src.tennis_scene.scripts.clip_studio \
  project_path=outputs/clip_studio/match1/project.json \
  recording_id=match1 \
  video_paths='[data/raw/cam0.mp4,data/raw/cam1.mp4]'

# 再開
.venv/bin/python -m src.tennis_scene.scripts.clip_studio \
  project_path=outputs/clip_studio/match1/project.json

# ヘッドレスエクスポート
.venv/bin/python -m src.tennis_scene.scripts.export_clips \
  project_path=outputs/clip_studio/match1/project.json
```

キー割り当ては GUI 内で `h`。同期は `a`（音声自動同期）+ `[`/`]`（選択カメラのフレーム単位ナッジ）。

## エクスポート形式

`<dataset_root>/dataset.json` に全クリップのインデックスを持ち、各クリップは `clips/<recording_id>/<clip_name>/` に `clip.json` と `media/<camera_id>.mp4` を持つ。これにより別収録で `clip_000` が再利用されても衝突しない。マニフェストの `video_paths` / `camera_ids` はそのまま `run_pipeline.py` に渡せる。

fps・解像度がソース間で異なる場合は明示的な `export.fps` / `export.width+height`（letterbox）を要求し、暗黙のフォールバックはしない。書き出した動画は re-probe して fps / フレーム数 / 解像度をプランと照合し、違反があれば例外を投げる。疑似アノテーションの追加方法は `../generate_dataset/README.md` を参照。
