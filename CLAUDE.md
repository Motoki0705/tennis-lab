@AGENTS.md

<!--
正本は AGENTS.md と .agents/（Claude・Codex 共通）。Claude Code は @path import で
AGENTS.md を読み込む。ここには Claude Code 固有の差分だけを書く。
-->

## Claude Code 固有

- auto-memory（`~/.claude/projects/<project>/memory/`）には、ユーザー個人の嗜好と個人の環境だけを保存する。プロジェクトの知見は、Codexからも読めるように共有memory [.agents/memory/](.agents/memory/README.md) に書く。
