# `/goal` prompt

repository rootで次をメインエージェントへ送る。

```text
/goal GitHub Issue #753を完了する。最初にcleanな`main`上で`.agents/skills/issue-subagent-workflow`を初期化し、失敗済み`.codex/tasks/issue-753`が残る場合だけupstream Issue更新後の`--refresh-issue`で再生成して`base_revision`をmainに固定する。その後`feat/blcs-track-query-cswa`をcheckoutし、`src/tasks/blcs/docs/track_query_cswa/`の順序で各production componentを独立Implementerへ委譲する。全spawnは`fork_turns="none"`とspawn contractの完全一致するterminal-only footerを使う。PR #755はユーザー許可済みの早期Draftであり完了根拠にしない。CUDA packageはrepo-root共有training queue経由で一つずつ直列実行する。subagent、CI、queue、長時間processは完了通知または最大timeoutのblocking waitで待ち、`list_agents`、短時間wait、status・log・GPUの反復確認をしない。BLOCKEDで停止し、最終PR headの`capture-pr`、required checks、`finalize-pr`、最終workflow checkが成功するまで継続する。
```
