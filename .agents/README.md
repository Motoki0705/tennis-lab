# AI運用の正本

Claude・Codexなど、このrepoで働くAIエージェントの運用ルールの正本。入口は [AGENTS.md](../AGENTS.md)。どのツールも同じ規則で動き、ツール固有のファイルには差分だけを置く。

設計原則・GitHub運用ルール・研究ループは、#1053 本文から移した（#1056）。今後はこのファイルを改訂し、#1053 本文は更新しない。

## 設計原則

1. **固めすぎない**: ルールは仮説として置く。運用データ（事故・手戻り・放置）をもとに週次で改訂する。AIの制御（このファイル、skill、memory、自動運用）自体も、プロジェクトと一緒に最適化する対象である。
2. **正本は1つ**: Claude と Codex は対等。`AGENTS.md` と `.agents/` を共通の正本とする。ツール固有のファイル（`CLAUDE.md`, `.github/copilot-instructions.md`, `.github/agents/`）には差分だけを書く。
3. **memoryは消去可能**: memoryは規則ではなく、検証待ちの観察である。古いもの・誤っているもの・固定観念になったものは安く消す（git履歴が保険）。詳細は [memory/README.md](memory/README.md)。
4. **既定コンテキストは小さく**: 常に読み込むのは、汎用で頻繁に使うskillだけにする。大きなskill群（例: ARIS）は既定のskill置き場に置かず、必要なときに明示的に読み込む。
5. **破壊的変更は積極的に**: 後方互換は保たず、旧経路は同じPRで削除する。ただし共有基盤を壊す変更は人間が承認する。どのパスが共有基盤かは [review/README.md](ops/review/README.md) を参照（#1058で実装中）。

## 読む順序

1. [AGENTS.md](../AGENTS.md): プロジェクト概要と開発ルール（全セッション）
2. このファイル: 運用の原則と、情報の置き場所
3. [memory/INDEX.md](memory/INDEX.md): 共有memoryの索引。作業に関係するエントリだけを開く
4. 作業対象ディレクトリの `README.md`（大きなディレクトリには必ずある）

## ファイル地図（情報の住み分け）

同じ事柄は1か所にだけ書く。書く前に、下の表で置き場所を決める。

| 置き場所 | 書くもの | 書かないもの |
|---|---|---|
| [AGENTS.md](../AGENTS.md) | プロジェクト概要、全作業に効く開発ルール | 手順の詳細、観察、実験結果 |
| `.agents/README.md`（このファイル） | AI運用の原則、置き場所の地図、GitHub運用、研究ループの概要 | 各自動運用の実装詳細 |
| [.agents/skills/](skills/) | 繰り返し使う作業手順（`SKILL.md` + scripts） | 一度きりの経緯 |
| [.agents/memory/](memory/README.md) | まだ規則でも仕様でもない、再利用できる観察（落とし穴、環境の癖、ユーザー判断） | コードやREADMEで確認できる事実、作業の進捗 |
| [.agents/ops/](#自動運用) | 定期実行する自動運用（systemd --user timer、headless agent） | 対話作業の手順 |
| [knowledge/](../knowledge/README.md) | 実験結果・考察・関連論文（1 run = 1ノード） | 運用ルール |
| 各ディレクトリの `README.md` | モジュールの仕様・使い方・設計判断 | 横断的な運用ルール |
| GitHub issue / PR | 作業の目的・進捗・決定の経緯 | 恒久的な規則（決まったらここか AGENTS.md へ移す） |
| 各ツールのローカルmemory（Claude auto-memory など） | ユーザー個人の嗜好、個人の環境 | プロジェクトの知見（共有memoryへ） |

ツール固有のファイル:

- `CLAUDE.md`: `@AGENTS.md` を読み込むだけにする。`.claude/skills` は `.agents/skills` へのsymlink。
- `.codex/`: Codexの設定と参考資料。
- `.github/copilot-instructions.md`, `.github/agents/`: Copilot固有の承認フローだけを書く。

## 自動運用

自動運用は、kamimura ユーザーの systemd --user timer で動かす（self-hosted runner は隔離されているため使わない）。timer の有効化は、PRのmerge後に人間が行う。

| パス | 内容 |
|---|---|
| [ops/lib/agent_exec.sh](ops/lib/agent_exec.sh) | `claude -p` / `codex exec` を read-only または write モードで headless 起動する共通wrapper。失敗や空の出力は非ゼロで終了する |
| [ops/install_units.sh](ops/install_units.sh) | `ops/*/systemd/` の unit を `~/.config/systemd/user` へリンクする。`--enable` を付けたときだけ timer を有効化する |
| [ops/weekly-report/](ops/weekly-report/README.md) | 週次の運用レポートissue（逆提案）。#1057で実装中 |
| [ops/review/](ops/review/README.md) | クロスレビュー、merge gate、PR作成agentの判別規約、共有基盤パスの一覧。#1058で実装中 |
| [ops/cleanup/](ops/cleanup/README.md) | merge済みbranch・worktreeの自動掃除と、承認制の削除候補一覧。#1060で実装中 |

利用枠の不足などで自動運用が失敗したときは、黙ってskipせず記録する。

## GitHub運用ルール（初版）

- **issue起票**: AIは、作業中に見つけたバグ・負債・派生課題を自由に起票してよい。AIが起票したissueには `ai-proposed` ラベルを付ける。週次レポートで人間が整理する。
- **担当の排他**: 仕組みは作らない（衝突は今の痛みではない）。問題が観測されたら週次レポートで見直す。
- **放置PR**: N日動きのないPRは、週次レポートに「継続 / rebaseして仕上げ / close」の選択肢付きで載せる。
- **close**: 実装issueは、PRのmerge（`Closes #`）で自動closeする。研究テーマのissueは、収束後に人間がレポートを読んでcloseする。
- **merge**: CI通過とクロスレビュー（Claude作成のPRはCodexが、Codex作成のPRはClaudeがレビューする）を条件とする。タスク内に閉じた変更は自動mergeしてよい。共有基盤に触れる変更は人間が承認する。判定の詳細は [review/README.md](ops/review/README.md)（#1058で実装中）。

## 研究ループ（概要）

- **単位**: 研究テーマが親issue、1サイクル（調査→選定→実験→考察）が子issue。実測した結果は `knowledge/` のノードに、次サイクルの提案は子issueの結論に書く。
- **人間が介入するのはテーマ設定だけ**: 問題提起・目標指標・評価条件・収束/停止条件を人間とAIが一緒に決め、テーマissueに固定する。以降の手法調査・選定・実験・次の調査は、AIが自律で回す。途中経過は週次レポートで観察する。
- **収束判定**: 共通のルールは持たない。テーマ設定時に、AIが指標の性質に応じた閾値・seed数・停止条件を提案し、合意して固定する。
- **ARIS**: そのまま導入しない。training-queue と knowledge に合わせた軽い自前skillとして作り直し、ARISからは有用な部品だけを取り込む。既定のskill群は肥大させない。
- 手順の正本: [skills/research-loop/SKILL.md](skills/research-loop/SKILL.md)（#1059で実装中）

## このファイルの改訂

- ルールは仮説である。合わなくなったら、週次レポートの提案として改訂する。
- `AGENTS.md` と `.agents/**` は共有基盤なので、変更するPRは人間が承認する。
- 規則を追加する前に、memoryの観察で済まないか、既存の規則を消せないかを先に検討する。
