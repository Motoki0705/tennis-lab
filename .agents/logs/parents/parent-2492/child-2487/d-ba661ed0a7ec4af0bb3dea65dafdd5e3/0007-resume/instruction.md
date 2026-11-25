# #2487 統合担当: DINO＋ViTPose-H の一体速度を修正版harnessで測る（親の判断）

前回の報告（`c7c76e32a`）を受け取った。取込、精度の記録、#1739 の制約の記載を承認する。

## 判断: 一体速度を追加で測る
- ECPose は検出と姿勢を1つのモデルで出す。比べる相手の現行経路は、DINO で人物を検出し、その crop を ViTPose-H に渡す2段の経路。2段の時間は、crop の数と段の間の処理で変わる。DINO 単体と ViTPose 単体の値を足しても同じにはならない。だから現行の一体速度は、トレードオフ図に必要な基準になる。
- PT32 は測らない（ユーザー判断）。PT16 と TRT16 だけを、main の修正版 harness（`59faa32a2` を含む自分の HEAD）で測る。parity は `parity_not_evaluated_reason` で明示的に未判定にし、表では「PT32未計測」とする。
- 条件は、初回の暫定記録で DINO＋ViTPose-H を一体で測ったときと同じ構成にする。入力は固定 frame、batch size も同じ。ViTPose の flip の有無は、現行の標準 pipeline の設定に合わせる。crop を作る規則（DINO のスコア閾値、人数の上限など）も標準 pipeline と同じにし、記録に書く。
- 新しい出力先を使う。GPU は training queue 経由。小さい job にまとめる。
- 初回の PT32 暫定値（590.33 ms/画像）は、修正前の harness の値として残す。新しい値と並べるときは、harness が違うことを書く。

## 終わり方
テストと pre-commit を通して push する。最終回答には、測る条件、入れた job、push した HEAD を書き、queue を待つ `PARENT_RESULT`（`job_ids` 付き）を返す。
