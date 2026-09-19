# SLCS PR clip

既存の `overlay_2d.mp4` と `scene_3d.mp4` をCPUだけで合成する。推論は行わない。
出力は `outputs/slcs/visualize/<experiment>/<run-id>/` の `comparison.mp4`、
先頭・中央・末尾の固定3時刻の `contact_sheet.png`、`provenance.json`。
既存runへの上書きは禁止。失敗時の部分出力も残すため、新しいrun-idで再実行する。

```bash
.venv/bin/python -m scripts.visualization.slcs_pr_clip \
  --overlay /absolute/source/overlay_2d.mp4 \
  --scene /absolute/source/scene_3d.mp4 \
  --experiment pr_baseline --run-id clip-001 \
  --label 'Meiji / real RGB' --model baseline --epoch 56 \
  --clip-id video_001/clip_000 --camera-id cam0 \
  --checkpoint-sha256 CHECKPOINT_SHA256 \
  --start 3 --end 11 --fps 10
```

区間は事前に固定して明示指定する。性能による選別は行わない。
入力は時刻0から始まるCFR動画で、末尾サンプル時刻の差は3D動画の1フレーム以内とする。
時刻 `t` では各動画の `floor(t * source_fps)` フレームを表示するため、
3D動画が3フレームおきの低FPSでも実時間が対応する。元動画同士が時刻0で同期済みであることは呼び出し側の責任。
全画角を維持し、拡縮と余白追加のみを行う。補間・平滑化・外れ値除去は行わない。
2Dのボール影はz=0の地面投影であり、真の3D再投影ではない。
teacherはpseudo-3Dであり、実測ground truthではない。

モデル名・epoch・clip・camera・checkpointハッシュは呼び出し側から明示的に渡す。
provenanceには入力動画のSHA-256、実FPS、フレーム数、抽出区間、出力FPS、
コマンド、採用時刻を記録する。チェックポイント本体は読み込まないため、
指定したハッシュと学習メタデータの対応は呼び出し側で確認する。
必要環境: ffmpeg/ffprobe、Pillow、DejaVuSansフォント（Ubuntu標準パス）。
デコード・エンコードは1スレッドで逐次処理し、フレーム列全体は保持しない。
