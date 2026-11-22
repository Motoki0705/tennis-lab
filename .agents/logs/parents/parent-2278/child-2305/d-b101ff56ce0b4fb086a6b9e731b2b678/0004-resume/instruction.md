# 続き（同じsession・branch・worktree）

前回のターンで投入したジョブ（`i2305-*-r3` の4件）はすべて終了状態になった。結果は各ジョブのログ・出力で確認すること（成功とは限らない）。

以後のCI・queue待ちは共通ワーカー指示の「CI・ローカルqueueの受け渡し」に従い、最終メッセージ末尾の `PARENT_RESULT` で待機対象を宣言して終了する（parent CLIが結果を受けて同じsessionを再開する）。自分で待たない。当初の時間上限・止めてよい条件とAcceptanceは引き続き有効。Acceptanceをすべて満たしたら、PRを作成・pushし、CI待ちを `resume_on_success: true`（next_step: ドラフト解除と最終報告）で宣言する。
