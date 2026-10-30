@AGENTS.md

<!--
正本は AGENTS.md と .agents/（Claude・Codex 共通）。Claude Code は @path import で
AGENTS.md を読み込む。ここには Claude Code 固有の差分だけを書く。
-->

## Claude Code 固有

- auto-memory（`~/.claude/projects/<project>/memory/`）には、ユーザー個人の環境（パス、アカウント、マシン固有の設定）と、Claude Code の使い方に閉じた好みだけを保存する。プロジェクトの知見とユーザーの判断は、Codexからも読めるように共有memory（[memory skill](.agents/skills/memory/SKILL.md)）に書く。
- 各セッションの最終回答は日本語で行う。
