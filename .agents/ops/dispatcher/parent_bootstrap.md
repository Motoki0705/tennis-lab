# 自律親の起動

末尾の `bootstrap_input` の親番号、repo、stateの場所、候補と根拠を確認する。
[parent-session](../../skills/parent-session/SKILL.md) の「dispatcherから起動された」の行を読み、その参照先に従ってこの予約を受理する。
待機・終了時は同じ親の `workflow-report` の出力を最終文にする。bootstrapのprivateパス・session IDは公開しない。
