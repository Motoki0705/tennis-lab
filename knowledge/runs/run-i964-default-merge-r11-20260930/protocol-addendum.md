# Run 11 addendum: 既定追跡の共通化と重複person boxの固定比較

2026-09-30。[ユーザー決定](https://github.com/Motoki0705/tennis-lab/issues/964#issuecomment-5908318161)に従う。
この文書をcommit/pushしてIssueへ投稿した後に、新条件の実データ追跡・採点を行う。
[run 8 protocol](../run-i964-tracker-matrix-r8-20260930/protocol.md)、
[run 9 addendum](../run-i964-tracker-linking-r9-20260930/protocol-addendum.md)、
[run 10 addendum](../run-i964-tracker-hybrids-r10-20260930/protocol-addendum.md)の指標・照合・選別を継承する。

## 変更と固定条件

- 標準pipelineと学習文脈用の共通入口の既定をStrongSORT++＋pose/CLIPへ変更する。
  pose重み.15、AFLink/GSIを含むrun 10設定を維持し、他方式は明示指定のみ。
  AFLinkは論文再実装と公開重み（SHA-256
  `b35cbeddd3acc48fece820bd640640e6bfb1f5fbf570aa79af26c6a38958daa4`）を使用する。
  重みの独立した利用条件は未確認で、READMEとknowledgeへ記録する。重みは再配布しない。
- **merge off**と**merge on（IoU >= .8）**の2条件だけを比較する。既定はoffのまま。
  各frameのperson検出をscore降順、同点は元row昇順に処理する。
  未処理boxから最上位を残し、残したboxとのIoUが.8以上の残りを落とすgreedy方式。
  dropped boxを経由した推移的な連結はしない。class情報は使わない。
  出力は元row順へ戻し、元box/scoreを変更しない。各dropにframe、kept/dropped元row、両score、IoUを保存する。
- COCO全画面.30、800/1333、4 dev clip×3camera、保存済み40,531検出row/ViTPose/CLIPを使う。
  mergeは保存特徴の同じrowを間引く。保持rowのpose/外観を再計算しない。
  検出器・閾値・固定コート座標選別・上限6（選別後）・v3安全策は変更しない。
  ラベルをmerge/追跡/選別へ渡さず、閾値探索はしない。予約未見は開かない。
- 元検出rowを全段階で維持する。GSIは別box配列・synthetic maskとして保存し、
  実観測box/pose、選別、raw/group IDF1、pair F1へ昇格させない。
  両条件を共通production入口で実行し、offはrun 10のtrack ID/box/rowおよび指標と照合する。

## 報告

- raw IDF1（主）、group IDF1（追加副指標）、CLIP camera間pair F1と決定clip数/coverage、
  switch/fragment、選手保持、既知非選手残存、unknown/未照合、camera×near/farを同じ母数で示す。
  停止は空予測/FNとして計上する。
- clip/camera別に元box数、drop数、変更frame数を保存する。全merge記録について
  元の部分参照とのIoU>=.5照合で、keep/dropが別identityの人物を最良照合する事例を列挙する。
  同じidentity以外をdropだけが覆う場合も列挙し、実画像と前後frameで確認する。
  部分参照にない別人はこの検査では否定できないため、既知の別人削除と未確認を区別する。
- 3camera上下（off/on）の短い動画を作る。各clipでmergeが生じたframeの5秒窓のうち、
  dropped box数が最多の窓（同数なら最も早い窓）を選ぶ。merge無しclipは省く。
  kept/drop box、raw/group ID、GSI syntheticを区別し、全frame読戻しとSHA-256を記録する。
  意図せず別人を落とした候補があれば、そのframeを追加の静止画/動画で示す。

## 資源と終了条件

CPUのみ、最大4thread（BLAS/OpenCV1）、pytest -n4、RAM available >=6GiB、追加disk <=5GB。
見積り: 共通化/テスト/資料2–4時間、2条件CPU追跡・採点20–45分、動画10分、出力<=300MB。
GPU/queueは0件。#935/#936のbranch/worktreeは変更しない。
今回は比較結果からmergeの既定や閾値を変えない。残りの#933再較正、clip_000全pipeline、
調整凍結後の未見1回評価は実行条件・予算・ノイズへの過適合を避ける手順を文書化するだけ。
