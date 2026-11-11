## 親からの継続指示（子3: 調査と設計の報告の続き）

前のターンは、このラップトップのメモリ不足で、親側の待機プロセスと一緒に停止した。これはworkerの作業の問題ではない。ここまでに調べた内容（events・会話の履歴）を引き継ぎ、前回の指示（下に再掲）にある報告1〜6を完成させて返す。このターンでは引き続き実装しない。

- 調べ直しは、前回の結果を確認できない箇所だけにする。
- pytestは原則実行しない。必要な場合も `-n 0` で1〜2 nodeまでとする。メモリを大きく使うコマンド（全体のgrep・大きなファイルの一括読み込みなど）は避ける。
- 目安は最大45分。

---

## 親からの具体指示（子3: 正式レビューのモデル選択 — 今回は調査と設計の報告）

子1（#1860、PR #1867）と子2（#1861、PR #1872）はmainへmerge済み。
- 子1: 保存済みfinalの検証付き採用（recover）、公開再開、PRごとの証拠待ち。
- 子2: 照合表示（`parent review-inspect`）、PR単位の排他、中断runの明示終結（`close-interrupted`）。

レビューの契約の正本は `.agents/ops/review/README.md`、共通ランナーの契約は `.agents/ops/lib/README.md`。

このターンでは実装せず、次を調べて報告する。

1. 現状の固定経路を整理する。`agents.conf` の `REVIEW_CODEX_*`（現行は gpt-6-astra / xhigh）が、`agent_exec.sh` のreview profile、`parent_review.py run`、実効設定の照合・保存（agent-runtime）、`review_recovery`（recover時の元run検証）、merge gate・公開記録の `reviewer` / `profile` 欄へ、どう流れているか。review profileでモデル指定を拒否している箇所を特定する。
2. 選択値の固定場所を設計する。手動親（`parent review-request`）と自律親（dispatcher等の入口）の両方から、同じ経路でモデル・effort・選択理由を渡す。人間の明示指定を優先する。選択を request（digestに含まれる）と execution に固定する。次も示す。
   - 許可リストと導入モード（fixed / select、既定fixed）を `agents.conf` にどう表すか。
   - fixedのときに選択を受け取った場合の扱い（拒否か、fixedの値と一致する場合だけ許すか）。
   - 旧request（選択欄なし）を読み取るときの扱い。
3. 実効値の照合を設計する。共通ランナーが記録する実効model・effortと、選択値の照合。照合はどこで行い、不一致・利用不能のときにどう失敗させるか（同じrunで別モデルへ黙って切り替えない）。recover・resume-publication・review-inspect・merge gateが、選択値と実効値をどう検証・表示するか。
4. 別runへの引継ぎを設計する。例: Solで未解決の範囲をAstraで追加レビューする。これは親の明示判断とし、前runとの対応を記録する。既存の `superseding_reviews`・後続requestの理由（exception）・PR排他と、どう整合させるか。
5. 自律親（`issue:parent` を自動で担当するdispatcher）がreview-requestを呼ぶ経路の有無と、そこに選択を渡すために必要な最小の変更。
6. 実装する場合の作業単位、変更ファイル、テスト設計、所要時間の見積もり。テスト設計には次を含める。
   - fake runtimeで、Sol/Astra・人間指定・fixed・selectの各場合と、実効値の不一致・利用不能・旧requestを確かめること。
   - 既定fixedで、既存のreview・gateテストが不変であることを確かめる方法。

制約: このマシンはRAM約3.9GBのラップトップ。pytestを実行する場合は `-n 0` で少数のnodeだけにする。実agent・実GitHubへの書込みはしない。`.agents/ops/lib` は下流が広いので、影響範囲を具体的に挙げる。調査の目安は最大1時間半。
