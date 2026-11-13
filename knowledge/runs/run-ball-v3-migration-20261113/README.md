# v2実データ移行の実行記録

対象issueは #1714。実行対象と入力版の初回観測は `selection.json`、注記を読んだ本実行集合は
`selection-reviewed.json`、実測は実行後の `results-summary.json`、一度きりの実行入口はこのディレクトリのscriptに固定する。
旧JSON、review、動画、既存frame storeは上書きしない。

## 実装前の検証設計

- 契約: 独立して固定した元動画identity・camera・収録group・実frame/PTSを照合し、
  v2の観測・補間・遮蔽推定・unresolvedを、v3の元座標と由来へ明示的に移す。
  targetと処理完了の根拠を必須にする。未知から不在や遮蔽を生成しない。
- 故障: 正規化の分母変更、推定の観測化、support欠落、同数frameのPTS差替え、
  全ボール教師のプレー球化、旧holdoutのtrain移動、再送の二重追加を検出する。
- 層: converterの異常分類はunit。移行済みJSON・prepared画像と既存publisher・snapshot・
  学習readerの接続はintegration。実データは少数dry-run後に固定集合だけを実行する。
- 再利用: `tests/unit/tasks/ball_detection/annotation/video/test_io.py::test_v2_correspondence_preserves_known_information_without_inventing_states`、
  `tests/integration/tasks/ball_detection/test_video_publication.py::test_publication_preserves_base_bytes_rows_and_sealed_review_masks`、
  `tests/integration/tennis_scene/test_ball_annotation_imports.py`。
  既存テストは削除せず、移行専用の入力根拠と拒否を追加する。
- コスト: 新規unitは30case以下、integrationは5case以下を目安にし、1case5秒・setup1秒を
  超える場合は実測と理由を記録する。小さなCPU合成動画を使い、実GPT・GPU・学習は行わない。
  指定の `pytest_locked.sh`（-n 4）で対象nodeだけを実行し、収集件数とdurationsを保存する。

## 棚卸し時の観測

Meijiは57clip・171camera動画・88,872frame。元動画は約1.56GB。
ローカル旧frame store 3版は全171本を同じgroup/splitで含む。
旧storeの画像は縮小済みのため、native画像への偽装や逆拡大は行わない。
全ボール形式のchat/TrackNet教師は別に記録し、自動的にactive_ballへ採用しない。

`all_frames_reviewed=false` は独立した目視の網羅性の申告であり、直ちに処理未完了と
解釈しない。処理完了の別根拠がない入力と、手持ち球を含む明示的な対象相違は保留する。


初回の132本選択は `selection.json` に保持する。注記を全件読んだ
`selection-reviewed.json` が本実行の123本・63,540frameを指定する。
手持ち・プレー前後の球を含む39本と、処理完了を確認するreportがない15本（6本重複）は
全cameraをまとめて保留した。全frame自動処理・decodeの明記がある入力は、個別目視の
非網羅性と分けて採用した。判断に使ったJSON pointerと原文は同一覧に保持する。
`review_selection.py` はこの具体的な原文にだけ適用し、一般の文章分類器にはしない。

Driveの場所・容量・旧/新readerは `storage-and-readers.json`、旧archiveの99件と
現行版との内容一致は `archive-inspection.json` にある。旧archiveにはreview/reportがなく、
保留対象の処理完了を補える新しい証拠は得られなかった。
少数公開は `meiji-active-ball-v3-dry-20261113`、本集合は `meiji-active-ball-v3-20261113`
という独立の新versionへ保存する。native JPEGはsnapshotへhardlinkする。


本実行の初回attemptは、8本の準備後、prepared JPEGの公開保護漏れを修正するために
SIGINTで停止した（終了コード130）。v3や旧ファイルを上書きせず、準備済み成果物を再利用する。
追加したGC回帰は修正前に参照数0で失敗した。既存receiptの不変な保護記録を変更せず、
manifest内の画像archiveへ追加の公開保護を作る修正を実装した。
