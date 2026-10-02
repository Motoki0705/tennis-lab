### 進捗ログ run 30 (2026-10-01) — Meiji producer実装 / 実行前登録
- 実施: [762e3e05f](https://github.com/Motoki0705/tennis-lab/commit/762e3e05f) をcommit/push。#964凍結記録の人物設定・資産hash・共有person codeを照合し、全画面COCO DINO → FeatureExtractor（ViTPose全17点＋CLIP）→共有track_sequence → frame0 court/共有選別cap6を接続。候補A設定も照合し、camera間associationは生成しない。
- cache: Meiji train/video_002＋val/video_000の108 camera-clip / 52,866 frame。未検出・選別なし・court失敗を各frameで区別。runtime失敗/中断のattemptとframe範囲を保持し、次の明示的な実行でそのclipだけ再開。完了clipの入力/JPEG/code/model/NPZ/hash不一致は停止。TrackNet/chatはabsent_by_policy。
- 検証: CPU pytest -n4でcache/再開/改変拒否32件、凍結/shared tracker/全17点/欠測/例外3件の計35 passed。ruff/mypy/commit hooks成功。GPU生成はまだ未実行。
- 予定job: 1件、推定8–12時間。外側timeout 43,190秒、内部監視43,170秒（合計12h以内）。stage分離、torch allocator7GiB、全GPU 9.5GB停止（grant10GB以内の余裕）、RAM空き6GiB未満で停止。既存推論時間に基づく推定で、新producerの実測ではない。主要配列は上限6で約70MB、attempt＋assemblyと診断を含めcache1GBを見込み、硬い監視上限10GB。その他新規出力は5GB以内。resource=allで#964/先行#936の後にFIFO登録する。
- Acceptance: producerと通常検証まで完了。context学習・ablation・clip_000 qualificationは未実施。GPU job IDと最終固定plan hashは投入時に追記する。
- 次の一手: 下記の事前規則でCPU benchとrun29の3件を解析し、cacheのCPU preflight/計画を固定後、1jobをenqueueしてWAITING_QUEUE。

【要判断】108 clip cacheとclip_000実行禁止の境界
- 前提情報: 承認済み108 camera-clipはvideo_000/clip_000の3cameraも含む。directiveは108clipのcontextと、clip_000 qualificationの次run提案を同時に指定している。
- 選択肢: A. 108clipの教師を使わない人物/court文脈生成のみ許可範囲と解釈 / B. clip_000を除いた105clipへ変更。
- 採用した暫定案と理由: A。明示された108clipの母数を守る。clip_000のball/refiner/court_side、qualification、閾値選定・精度評価は実行しない。video_001は対象外。
- 覆す場合の影響範囲: 生成計画の3clipとcache coverage。球モデル・confidence規則は変更しない。
