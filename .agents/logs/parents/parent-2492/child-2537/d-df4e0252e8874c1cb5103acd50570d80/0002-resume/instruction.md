# #2537 親レビュー結果: 2点直してから push してよい

原稿を読みました。構成・仕様・用途・リンクは承認します。次の2点を直し、検証をやり直してから Wiki に commit・push してください（push はこの指示で承認済み）。

## 1. 3cam と 4cam が別収録であることを結論として書く

「学習データとの重複」の SLCS の項は、SHA-256 一致0本と frame 数の違いまでで、「別の収録か」の結論がありません。ファイルが違うだけでは、同じ試合の再エンコードの可能性を否定できないためです。親が確認した事実:

- 元動画が別のファイル群: `projects.json` の sources が `tennis_multiview/raw/meiji_3cam/video_000/camN.mp4` と `tennis_multiview/raw/meiji_4cam/video_000/camN.mp4`。
- `ffprobe` の `format_tags=creation_time` が約4か月離れ、長さも cam0 で約508秒（3cam）と約1,980秒（4cam）で違う。

自分でも `ffprobe -v error -show_entries format=duration:format_tags=creation_time` で全カメラを確かめ、`tmp/wiki-evidence.md` に追記してください。そのうえで:

- SLCS の項に「SLCS 実RGB の `video_000` は meiji_3cam の別収録で、このベンチマーク（meiji_4cam）の映像は含まれない」と結論を書く。BLCS の旧Meiji重みとボール2D検出の既定重み（どちらも3camの `video_000`[local path]`video_002`）にも同じ結論を一言添える。
- 公開ページには収録日の具体的な日付を書かない（「作成日時が約4か月離れ、長さも異なる別収録」程度でよい）。
- 「同じ `video_000` という名前でも 3cam と 4cam は別の収録」という注意を `> [!WARNING]` か `> [!NOTE]` で1つ置く（検索の落とし穴になるため）。

## 2. 「各モデルでの利用可否」の冒頭に結論を2〜3行で置く

ユーザーの問いは「SLCSなどで使用可能か」なので、表の前に結論を短く書く。例:

- そのまま使えるのは、人物検出・2D姿勢の疑似GT採点と、元映像を入力にする標準パイプライン・単眼PLCS。
- SLCS・多視点PLCS・BLCS・ball_refiner_3d は、入力の組み立て（変換コードは main にない）が要る。
- 調べた学習経路（SLCS・PLCS・BLCS・ball_refiner_3d・ボール2D検出の既定重み）は、この4cam収録を学習に使っていない。ただし他の派生重みや事前学習データは未確認。

表と本文の内容は変えなくてよい。

## push の手順

1. 上の2点を直し、禁止項目の grep・リンク検査・数値照合をやり直す。
2. `git -C tmp/wiki pull --rebase` で Wiki の最新を取り込み、4ファイルを1 commit にまとめて push する。commit メッセージは日本語で既存に合わせ、`Co-Authored-By` などは付けない。
3. 公開URL `https://github.com/Motoki0705/tennis-lab/wiki/Meiji-4cam-Benchmarks` が取得でき、サイドバーにリンクが出ることを確かめる。
4. Wiki の commit hash と確認結果を最終回答に書き、`PARENT_RESULT {"schema_version":1,"state":"done","wait":null}` を返す。
