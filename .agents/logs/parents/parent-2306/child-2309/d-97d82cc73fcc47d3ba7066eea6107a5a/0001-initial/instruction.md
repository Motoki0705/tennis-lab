# #2309 調査ターン（実装しない）

担当issue #2309 の「具体的な変更内容・対象範囲」の先頭にある調査を行う。コード・文書は変更せず、commit・push・PRもしない。CI・学習の完了を待つ作業はないので、待機やポーリングはしない。

## 調べること

1. agy（`[local path]`、1.5.13）の対話セッションを一意に示す環境由来の識別子（`ANTIGRAVITY_CONVERSATION_ID` / `ANTIGRAVITY_TRAJECTORY_ID` 等）。会話の再開（`--continue` / `--conversation`）、`[local path]` 相当、サブエージェント・swarm・`run_workflow`・`-p` 実行・`agy acp` で値と親子関係がどう変わるか。実際に agy を `-p` などで起動し、hook や子プロセスから環境とプロセス木を観測して確かめる（課金・副作用のある操作は避ける）。
2. hook（`SessionStart` / `PreInvocation` / `Stop` 等）の設定場所（ユーザー・プロジェクト・plugin）と、hook が受け取るJSON入力（`hookEventName`、`cwd`、会話ID等）。新規・再開・クリアを区別できるか。repo既存の `.agents/ops/lib/agy_hook.py` 等との関係。
3. 対話セッションへ外部からメッセージを投入する手段（queued messages のファイル、`agy acp`、remote-control 等）の有無と、使う場合の信頼性・副作用。掲示板の呼び鈴として直接配送できるか、hook での未読提示に落とすべきかの推奨。
4. Herdr が agy の pane をどう認識するか（`pane["agent"] == "agy"`）と、既存の Claude / Codex 実装（`.agents/ops/lib/identity.py`、`.agents/ops/parent/`、`.agents/ops/herdr_labels/`、`.agents/ops/board/`）で agy 対応に必要な変更点の一覧（ファイルパス:行）。`allow_move` を Claude と同様に許可すべきかの判断材料。
5. 上記に基づく実装計画（変更ファイル、テスト方針、残リスク）。

## 報告

最終メッセージに 1〜5 を根拠（実測コマンドと出力、ファイルパス:行）付きで日本語でまとめる。確認できなかった点は推測で埋めずに明記する。作業時間の目安は最大90分。
