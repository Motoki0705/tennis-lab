## 親の具体的な指示（#1925）

- `register` / `handoff --accept` で、要約が保存されていない親に要約なしで実行した場合は、担当を書き込む前に拒否し、`--summary` の指定を求める。拒否時にregistry・issueコメント・表示が変わらないことをテストする。
- Herdrへの接続失敗など、要約以外の原因による表示同期の失敗は、従来どおり担当を適用し、終了値1・`display_sync.status=error` と修復手順の案内を返す。
- 掲示板の改名など、同じ要約の前提に依存する他の入口も確認し、扱いを報告する。
- 受入条件の「実機（Herdr）での確認記録」は、実在の親の表示を乱さない方法がある場合だけ行う。無ければ、merge後に親が行う確認手順を最終メッセージに書く。
- 並行作業: #1943（wait-worker）と #1704（parent-session skill の整理）も、`.agents/ops/parent/` と `.agents/skills/parent-session/SKILL.md` を編集する。自分の差分は担当範囲の箇所に留める。
- 作業時間の上限は4時間。止めてよい条件: 受入条件を満たしPRのCIが成功した。または、人間の判断が要る事項が残り、それ以外の独立した作業を終えた。

## この一括作業での共通の制約

- 親 #1968 は、元々 `issue:standalone` だった12件を同時に解消している。他に11のworkerが並行して動いている。
- pytestは変更に関係するnodeに絞り、`.agents/ops/pytest_locked.sh <対象> -q`（-n 4固定）で実行する。repo全体の一括実行はせず、全体はCIで確かめる。
- GPUを使う処理は training-queue skill 経由でだけ実行する。
- PRは担当issueを閉じる場合に `Closes #<担当issue>` を付け、本文は日本語で書く。PRのmerge、issueのclose、親issueへの投稿、systemctl・timerの操作は行わない。issueやPRへの公開の報告文が必要なら、最終メッセージに案文を書く。
- 最終メッセージは次の見出しで書く: `## 結果`（受入条件ごとの状況と証拠）、`## branch・commit・PR・CI`、`## issueへの報告案`（不要なら「なし」）、`## 親への判断依頼`（なければ「なし」）。
