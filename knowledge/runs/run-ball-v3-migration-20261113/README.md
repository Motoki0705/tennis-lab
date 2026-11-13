# v2実データ移行の実行記録

対象issueは #1714。実行対象と入力版は `selection.json`、実測は実行後の
`results.json`、一度きりの実行入口はこのディレクトリのscriptに固定する。
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
