# PR #2270 正式レビューのmajor指摘への対応（親の判断つき）

正式レビュー（parent-1734-f8284634df6c411287abf69ffa6df2d8）で、major指摘が3件出ました。親の判断は次のとおりです。再レビューは行いません。修正してCIが通ったら、親がmergeします。

## 指摘1: 直す（dataset publication）
`_required_scene_components` が証拠の種類を渡さずに `PinnedSideInputs.from_store(root)` を呼ぶ。そのため、ball段を外したstoreでは `ball_points/<camera>` がなくて `KeyError` になる。共有判定の証拠の種類を公開の検証に渡すこと。公開・再利用のテストを、ball成果物がまったくないplayer構成にも広げること。

## 指摘2: 直す（構成の判定）
- 両方の構成を、明示的に選べて、計画が一意に決まるようにする。
  - player証拠＋ball推論あり: #1757のbenchmarkで使う。
  - player証拠でballなし: 本番で使う。
- ball機能を無効にしているのに、ball段に個別に `execution.<ball段>=load` などを要求する設定は、計画を作る前に拒否する。
- 両方の構成の計画と、矛盾したときに止まることをテストする。

## 指摘3: 直さない（親の判断）
「割り当てを使わない集合間距離」は、親が指示を書いたときの表現です。本来の目的は、player_associationの結果（人物ID・外観）に依存しないこと、つまり循環を避けることでした。足元の点をフレームごとに幾何だけで一対一に対応させること（Hungarian法）は、人物IDを使わないので、この目的を満たします。採点の方法は変えなくてよい。

代わりに、READMEとPR本文に次のことを明記してください。
- Hungarian法は、各フレーム内で幾何だけを使う対応であること。
- player_associationの出力を使わないこと。

## そのほか
- #1757のworker（`codex/video-batch-benchmark`）が、このbranchを取り込んで使います。修正は早めにpushしてください。設定のキーが変わったら、PR本文の「#1757での使い方」を更新してください。
- 3clipの判定結果が変わらないことを、CPUで確かめられる範囲で確かめること。
- 最後に、HEAD・テスト・CIの結果を報告すること。
