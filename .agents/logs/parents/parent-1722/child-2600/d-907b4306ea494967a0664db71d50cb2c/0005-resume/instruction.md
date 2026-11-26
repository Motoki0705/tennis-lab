# PR #2613 正式レビュー（round 1）の major 修正と再実験

対象HEAD `7ae7fdc395caef5a39d1c29b2f514de685faaef3`。

親が行った変更を先に取り込むこと:
- #2613 の branch: 親が main を取り込んで `7ae7fdc3` にした（#2610 マージ後）。base は main に変更済み。
- 記録 PR #2621 の branch: 親が val の揺れについての段落を足し（`66ee82565`）、main を取り込んだ（`73059549`）。

Sol の正式レビューで major 2件が出た。親は2件とも採用する。

## 1. 判定不能区間に重なる確定・有力イベントが教師と採点から脱落する

`labels.py:132–140` は、`coverage=false` のフレームを BCE から除外すると同時に、そこに中心がある certain・probable の Gaussian 全体を生成していない。`scoring.py:58–65` も同じ条件で正例を落としている。

親の集計では、配布版 multiview-v3 の certain・probable 900件のうち15件が coverage=false のフレームにある。内訳は train 10（打球8・バウンド2）、val 4（ネット1・バウンド2・打球1）、test 1（打球1）。例として、train `clip_049` f700 の有力打球、val `clip_022` f1015 の確定ネット接触、test `clip_048` f367 の有力打球がある。

注釈者は判定不能区間の中でも、確かなイベントには印を付けている。したがって coverage=false は「その区間に印のない追加の接触があるかは判定できない」という意味で、印の付いた certain・probable は既知の正例である（人間の承認: certain と probable は正例、判定不能区間は loss から除外）。次のように直す。

- Gaussian は、coverage にかかわらず全ての certain・probable から生成する。BCE のマスクは、従来どおり coverage=false のフレームと uncertain の無視区間を除く。中心が coverage=false でも、裾が coverage=true のフレームにかかれば、そのフレームの教師値は Gaussian の値になる。
- 採点では、certain・probable は coverage にかかわらず照合対象の正例にする。coverage=false のフレームにある予測は、既知の正例と照合できれば TP、照合できなければ FP に数えない（判定不能）。uncertain の扱いは変えない。
- 両方の挙動を、実データの該当例に相当する小さな fixture でテストする。

## 2. Set Attention のテストが無効選手の key/value 混入を検出できない

`fusion.py:65–72` の SAB と pooling の両方から `key_padding_mask` を外しても、単体テスト7件が通る。`test_fusion.py:73–99` は無効枠を NaN にするが、attention の手前でゼロに置き換わるため、この回帰を検出できない。学習後相当の重み（ゼロ初期化の出力層も非ゼロにする）で、無効枠を含む入力と、無効枠を物理的に除いた選手集合の入力とで出力が一致することを比べるテストを足す。SAB と pooling のどちらか一方のマスクを外しただけでも、そのテストが失敗することを確かめる。

## 再実験と記録

- 修正後の HEAD で、主比較（3方式 × seed 42・43・44）と球のみ参照の計10ジョブを、同じ条件で training queue に投入し直す。これまでと同じく `TRAINING_QUEUE_DIR=[local path]` を使い、worktree から `add` する。前処理（`prepare`）の出力が変わる場合は作り直し、SHA-256 を記録する。全 job ID を `PARENT_RESULT` の wait で宣言して終了する。
- 結果を受け取ったターンで、#2621 の比較・表・図・条件別結果・本文を新しい結果に更新する。親が足した val の揺れの段落の数値（epoch 間の幅、選択 epoch、球のみとの差）も、新しい結果で計算し直す。結論が変わる場合は、変わった点を明記する。
- 旧結果（修正前）は、上書きせずに別の出力先に残す。記録では旧結果との差を1文で触れる程度にとどめる。
- #2613 と #2621 の PR 本文の実行証拠を、それぞれの最終 HEAD で更新する。#2621 の実行証拠の `result` には「未確認」「未検証」などの語を使わない（否定の文脈でも記録の gate が拒否する）。
