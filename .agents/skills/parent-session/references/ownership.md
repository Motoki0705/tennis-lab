# 担当と合意

1. `status` で担当・既存PR・ワーカー・停止事項を確認する。新規担当は `register --summary`、交代は前任の記録を読んで `handoff --accept`。要約は表示幅24桁以内で現在の仕事を書く。
2. `agree --spec` に目的・範囲・期待動作・受入条件・人間の元発言を保存する。既に明確な指示はそのまま使う。補足追加がpendingなら `wait-supplement` が対象runの確定を待つ。
3. `status.input_changes` で本文やコメントの変化を読む。範囲・受入条件・権限が変われば合意を更新し、合意内の順序・調査方法の変更は親が判断する。過去の証拠は影響を受けた部分だけ確認する。

### 引継ぎ

渡す親は `handoff --note-file` に進捗、未達、次の操作を書く。受ける親が同じIssueで受理すると既存の委任・PR・履歴を引き継ぐ。前任が予期せず止まった場合だけ、担当指定と停止状況を照合して `--accept --recover` を使う。

### 採用提案

レポート由来の作業は `receive-proposal` で元提案に結び付けてから合意する。終了時改善から取り出した親は `agree --improvement-approval` に人間の判断IDを渡す。dispatcherの自律根拠は [自律親](dispatch.md) に従う。
