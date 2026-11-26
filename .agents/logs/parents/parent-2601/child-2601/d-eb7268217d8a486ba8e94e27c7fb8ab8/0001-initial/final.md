#2601を実装し、通常PRを作成しました。CIは実行中です。[PR #2612](https://github.com/Motoki0705/tennis-lab/pull/2612)

- ブランチ `codex/parent-2601-issue-2601-primary`、コミット `6a7c5a82d`
- ブラウザで、再生中の得点入力、後続2件の再計算の取消・適用、再読み込み、改名・並べ替え、CSV入出力、消去、読取専用表示を確認しました。
- pytestは追加43件を含む249件、Nodeの再生回帰は10件成功しました。pre-commitも通過しました。
- 明治4カメラのv1は一時コピーで69クリップの読み込みとv2保存・再読込を確認しました。元ファイルのSHA-256は作業前後で一致しました。切り出し・動画書き出しは合成動画で確認しました。

CSV取り込み行は開始前カウント・サーブ側ともに手入力扱いです。硬式の第2セット以降から記録を始め、先行セットのゲーム数が不明な場合は、最初のサーブ側を手入力します。10点マッチタイブレークは最終セットを置き換える形式です。[ITF Rules of Tennis](https://www.world.tennis/media/7221/2026-rules-of-tennis-english.pdf)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2612,"head":"6a7c5a82ddf426a419a0706e02f5a0730642d3bf","resume_on_success":false}}