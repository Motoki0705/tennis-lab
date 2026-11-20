調査報告を承認する。実装へ進む（#2309 本文を更新済み）。

- 呼び鈴: 人間が「hook（PreInvocation）での未読提示」を選択した。内部RPCによる直接投入は実装しない（`[local path]` 後に旧会話を起こすため）。未読件数は検証済みルート会話の次の `PreInvocation` に渡し、本文取得と既読は `board` の操作で扱う。未読提示待ちを配送失敗と区別する。
- 識別: 報告の案（環境の会話ID・単一の前景TUI祖先・そのPIDが保持する会話ロックの照合、親CLIの入口で `rootConversationId`[local path]`trajectoryId` も照合して子履歴の開き直しを拒否）で進める。print・ACP・ネスト・サブエージェント・workflow は拒否する。terminal sandbox 内からの親操作は、ホスト側祖先を確認できないなら明示エラーで拒否してよい（受け付ける必要はない）。
- 表示: 報告の案（会話切替は statusLine/ロックの変化で検出、終了時に表示を消去、`allow_move` は新paneが対象会話のロックを保持し旧paneの空きを確認できた場合だけ）で進める。statusLine は起動単位の `--settings` で設定する案を採るなら、その起動方法を herdr-session-label 等の文書に書く。既存の読み取り専用 agy ランナーの設定検査は緩めない。
- 自律親での agy 親起動は対象外と文書に明記する。`agent:agy` ラベルと `agy/` ブランチprefixを追加する。
