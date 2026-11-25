# #2490 統合担当: PR #2500 の正式レビューの major 3件を直す

PR #2500（HEAD `a1a1fb742`）の正式レビューは major だった。精度18条件の再採点、速度30条件の再集計、RTMPose-X TRT16 の改善率、Pareto の判定は、掲載値と一致した。親は次の3件をすべて採用する。

## 1. 図と表に、精度を測った batch を明記する
Pareto 図のタイトルは「batch 1」で、CSV の batch 列は `batch_size` だけ。一方で平均 OKS は、全行で batch 8 の予測から取っている。README には結合の方法が書いてあるが、図と表だけを見た人には分からない。
- CSV に、速度・VRAM を測った batch と、精度を測った batch を別の列で持たせる（例: `speed_batch_size`、`accuracy_batch_size`）。
- 図のタイトルか注記に「速度・VRAM: batch N、精度: batch 8 の予測」と書く。batch 1・4 の精度は未計測であることが、図と表だけで分かるようにする。
- 生成コード: `knowledge/runs/run-meiji-topdown-pose-comparison/compare_speed.py` の341行目付近。

## 2. 比較 CLI で、レポートの差し替えと根拠のない欠測を拒否する
レビュワーの実行検証:
- RTMPose-X TRT16 batch 1 の元レポートをメモリ上で差し替え、summary の P50 だけを 1 ms に変えると、比較処理は新しい SHA-256 と 92.1% の短縮を出力した。
- 同じ条件を外し、`--omissions` に存在しないログを指定すると、`infeasible` として通った。

直すこと:
- 比較 CLI 自体で、各レポートを固定インデックス（`comparison_inputs.json`）の期待 SHA-256 と照合する。一致しなければ失敗させる。
- 欠測（infeasible）を受理する前に、対象条件の終端の証拠を確かめる。ログが存在し、その条件の失敗が記録されていること。満たさなければ失敗させる。
- それぞれ回帰テストを足す。差し替えたレポートと、存在しないログの欠測が拒否されることを確かめる。
- 該当コード: `compare_speed.py` の223行目付近。

## 3. RTMPose の精度ログを、queue の自動削除の対象外に保存する
RTMPose の精度6条件のログ参照は `.training_queue/logs/` を指している。このディレクトリのログは完了から7日で自動削除されるが、保持先の `repro/` にはログのコピーがない。ログが消えると、共通の再採点・可視化の読込（`replay_accuracy.py` の131行目付近）と、RTMPose 専用の再採点の読込がどちらも失敗する（レビュワーが消失を模擬して確認）。
- 6条件のログを保持先へ SHA-256 付きでコピーする。knowledge の参照と、両方の読込を、その保持先に揃える。
- ログは今は残っている。7日で消えるので、最初にコピーする。
- PMPose・ViTPose-H・Sapiens2 の記録にも、`.training_queue/logs/` を直接参照しているものがあれば、同じように直す。

## 終わり方
1. 各指摘の回帰テストを足す。関連テスト、pre-commit、knowledge 検査を通す。比較 CSV と図を再生成し、数値が変わらないことを確かめる。変わるのは列と注記だけのはず。
2. 同じ branch に push する。PR 本文の結果と検証の節を、新しい HEAD に合わせて直す。
3. 最終回答には、指摘ごとに直したこと（commit とテスト）を書く。新しい HEAD の CI を待つ `PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2500,"head":"<新HEAD>","resume_on_success":false}}` を返す。親が修正を照合して、マージ処理を続ける。
