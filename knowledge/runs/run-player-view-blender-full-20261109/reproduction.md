# C3 再生成用の固定入力とコマンド

人間の定性受入は未了。ここにある設定は初期案。

- clip: `clip000_20261103`、人物ID: `0`, `1`、元frame: `[0,1010)`、FPS: `59.94006`。
- ローカルの描画入力: `package-v2/manifest.json` と同じディレクトリの全ファイル。
- Driveの同一package: `outputs/tennis_scene/visualize/player_view/blender-acceptance-20261109/package`（Drive root相対）。
- package manifest SHA-256: `708affe5b630ba50014bf9f3f7a5ff4a730440906c1b1a4f99e920bfe5247280`。
- Blender source ref: `993c7e2310a76ebc7145ca33b9698be5b4996ddc`。
- Blender: `5.2.1 LTS`、Cycles / NVIDIA L4 / OPTIX、16 samples、persistent data有効、OpenImageDenoiseのGPU使用を明示。
- 保存cameraの解像度1280×720、水平画角90度、顔offset `(0,0.08,0.12)m`（head軸・person scale適用前）、near0.05m/far100m。
- 本人mesh全体非表示、相手灰色、球緑・半径0.05m、alpha1/emission0、簡素なコート/ネット。
- cameraは保存列を読み、人物/球/cameraの補間・Bezier・motion blur・reveal・fadeは使わない。
- 元frameとcache/Blender frameの対応はpackageの`source_frames.npy`。欠測でも時間を詰めない。

ローカルPCでGPU描画せず、Colabの所有sessionで資源手順に従って実行する。
新規の出力先を使い、GPUジョブの起動・安定確認は共通ランナーのCodex子に委任する。
queueは`resource=all`を使用する。新たな実行予算はその起動時に別途明示する。
以下はそのqueueジョブ内の実コマンドである（packageはDriveから取得済みとする。Colabの`C3_PACKAGE`は`outputs/tennis_scene/visualize/player_view/blender-acceptance/package`、ローカルで保持する同一入力は`package-v2`である）。

```bash
set -euo pipefail
ROOT=outputs/tennis_scene/visualize/player_view/blender-acceptance
: "${BLENDER_BIN:?Set the Blender 5.2.1 LTS executable}"
: "${C3_PACKAGE:?Set the verified package directory}"
test "$(sha256sum "$C3_PACKAGE/manifest.json" | cut -d " " -f 1)" = 708affe5b630ba50014bf9f3f7a5ff4a730440906c1b1a4f99e920bfe5247280
test "$(git rev-parse HEAD)" = 993c7e2310a76ebc7145ca33b9698be5b4996ddc
for identity in 0 1; do
  "$BLENDER_BIN" --background --factory-startup --python-exit-code 1 --threads 4 \
    --python src/tennis_scene/visualization/blender/shots/player_view.py -- \
    --package "$C3_PACKAGE" --output "$ROOT/reproduced/renders/player_$identity" \
    --player-id "$identity" --gpu-backend OPTIX
done
.venv/bin/python -m src.tennis_scene.scripts.publish_player_views \
  --package "$C3_PACKAGE" --renders "$ROOT/reproduced/renders" \
  --output "$ROOT/reproduced/videos"
for identity in 0 1; do
  cp "$ROOT/reproduced/renders/player_$identity/render.json" \
    "$ROOT/reproduced/videos/render_player_$identity.json"
done
ffmpeg -v error -n -i "$ROOT/reproduced/videos/comparison.mp4" \
  -frames:v 240 -an -c:v libx264 -threads 1 -crf 18 -pix_fmt yuv420p \
  -movflags +faststart "$ROOT/reproduced/videos/comparison_short.mp4"
```

最初の少数frame確認は各Blenderコマンドに
`--source-frames 16 17 18 19 730 731 732 733 734 735 736 737`を付け、
別の新規出力先を使う。公開時は`--preview`を指定する。
本番はframe選択を付けず、両人とも全1010frameを生成する。
`publish_player_views`は全frameのdecode・FPS・長さ・PTSを検査し、manifestへ結果を保存する。

実際の納品で使用した旧prefixと新segmentの出所は、動画と同梱する描画記録・checkpoint manifestを参照する。
この再生成コマンドは全frameを同じ最終rendererで新しく描き直すもので、過去の中断jobを再起動しない。
歴史的queue scriptの期限・元sourceはknowledgeのrun bundleへ原文のまま保持する。
