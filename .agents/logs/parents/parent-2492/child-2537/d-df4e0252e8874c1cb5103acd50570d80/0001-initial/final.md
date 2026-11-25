#2537 のWiki原稿を作成し、親レビュー待ちです。Wikiの `master`（基準 `fa7004c`）に未コミットの差分を置いています。

- 新規ページは217行です。概要と系譜、ディレクトリ構成、データ仕様、モデル別の利用可否と学習データの重複、用途、出典を収録しました。[Meiji-4cam-Benchmarks.md](https://github.com/Motoki0705/tennis-lab/blob/d08f191cc66ac3a50980bcd2a19de4f7c40d2551/tmp/wiki/Meiji-4cam-Benchmarks.md)
- VIのサイドバーと入口にリンクを追加し、HomeのVIを5ページから6ページに更新しました。[_Sidebar.md](https://github.com/Motoki0705/tennis-lab/blob/d08f191cc66ac3a50980bcd2a19de4f7c40d2551/tmp/wiki/_Sidebar.md)・[Research-and-Infra-Overview.md](https://github.com/Motoki0705/tennis-lab/blob/d08f191cc66ac3a50980bcd2a19de4f7c40d2551/tmp/wiki/Research-and-Infra-Overview.md)・[Home.md](https://github.com/Motoki0705/tennis-lab/blob/d08f191cc66ac3a50980bcd2a19de4f7c40d2551/tmp/wiki/Home.md)

人物検出・2D姿勢は既存の疑似GT採点器、HMR系は固定入力での定性・速度比較に使えます。標準パイプラインと単眼PLCSには元映像を渡せます。多視点PLCS・BLCS・3D Refinerは既存の推論API向けに入力を組み立てる必要があります。SLCSは構造化データ、DINOv3特徴、4人の収録と既存2人設定の対応が必要です。ボール2D検出は元映像で推論でき、採点にはrev1の確認済み2D点を使います。

SLCS実RGBの学習に入った `video_000` は3カメラ収録でした。学習時の映像39本と4カメラ版の276本をSHA-256で照合し、一致は0本でした。派生checkpoint、Chat由来の映像、別エンコードを含む重複は未確認として明記しました。

3版の実ファイルとmanifest、v2の人物配列の一致、人物評価v1の全66件のチェックサムと `main` での6,844画像の読込を確認しました。Wiki内リンク、`main` の参照先32件、Mermaid構文、公開禁止項目の検査も通過しています。v1・v2のチェックサム再計算はそれぞれ302件・68件です。[wiki-evidence.md](https://github.com/Motoki0705/tennis-lab/blob/d08f191cc66ac3a50980bcd2a19de4f7c40d2551/tmp/wiki-evidence.md)

親の承認後にWikiへcommit・pushし、公開URLとcommit hashを確認します。

PARENT_RESULT {"schema_version":1,"state":"needs_parent","wait":null}