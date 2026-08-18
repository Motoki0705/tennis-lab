# `/goal` prompt

設計PRを `main` へmergeした後、repository rootで次をメインエージェントへ送る。

```text
/goal GitHub Issue #753を `.agents/skills/issue-subagent-workflow` と `src/tasks/blcs/docs/track_query_cswa/` に従って完了する。各production componentは設計順に独立Implementerへ委譲し、全spawnは `fork_turns="none"` とterminal-only footerを使う。CUDA packageはrepo-rootのtraining queue経由で一つずつ直列実行する。subagent、CI、queue、長時間processは完了通知または最大timeoutのblocking waitで待ち、`list_agents`、短時間wait、status・log・GPUの反復確認をしない。BLOCKEDで停止し、`finalize-pr` と最終workflow checkが成功するまで継続する。
```
