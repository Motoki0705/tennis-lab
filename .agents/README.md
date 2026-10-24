# AI運用の正本

Claude・Codexなど、このrepoで働くAIエージェントの運用ルールの正本。入口は [AGENTS.md](../AGENTS.md)。どのツールも同じ規則で動き、ツール固有のファイルには差分だけを置く。

設計原則・GitHub運用ルール・研究ループは、#1053 本文から移した（#1056）。今後はこのファイルを改訂し、#1053 本文は更新しない。

## 設計原則

1. **固めすぎない**: ルールは仮説として置く。運用データ（事故・手戻り・放置）をもとに定期的に改訂する。AIの制御（このファイル、skill、memory、自動運用）自体も、プロジェクトと一緒に最適化する対象である。
2. **正本は1つ**: Claude と Codex は対等。`AGENTS.md` と `.agents/` を共通の正本とする。ツール固有のファイル（`CLAUDE.md`, `.github/copilot-instructions.md`, `.github/agents/`）には差分だけを書く。
3. **memoryは消去可能**: memoryは規則ではなく、検証待ちの観察である。古いもの・誤っているもの・固定観念になったものは安く消す（git履歴が保険）。詳細は [memory skill](skills/memory/SKILL.md)。
4. **既定コンテキストは小さく**: 常に読み込むのは、汎用で頻繁に使うskillだけにする。大きなskill群（例: ARIS）は既定のskill置き場に置かず、必要なときに明示的に読み込む。
5. **破壊的変更は積極的に**: 後方互換は保たず、旧経路は同じPRで削除する。ただし共有基盤を壊す変更は人間が承認する。どのパスが共有基盤かは [review/README.md](ops/review/README.md) を参照。

## 読む順序

1. [memory/SUMMARY.md](memory/SUMMARY.md): 共有memoryの要約。作業を始める前に必ず読み、詳細は層の索引から辿る
2. [AGENTS.md](../AGENTS.md): プロジェクト概要と開発ルール（全セッション）
3. このファイル: 運用の原則と、情報の置き場所
4. 作業対象ディレクトリの `README.md`（大きなディレクトリには必ずある）

## ファイル地図（情報の住み分け）

同じ事柄は1か所にだけ書く。書く前に、下の表で置き場所を決める。

| 置き場所 | 書くもの | 書かないもの |
|---|---|---|
| [AGENTS.md](../AGENTS.md) | プロジェクト概要、全作業に効く開発ルール | 手順の詳細、観察、実験結果 |
| `.agents/README.md`（このファイル） | AI運用の原則、置き場所の地図、GitHub運用、研究ループの概要 | 各自動運用の実装詳細 |
| [.agents/skills/](skills/) | 繰り返し使う作業手順（`SKILL.md` + scripts） | 一度きりの経緯 |
| [parent-board skill](skills/parent-board/SKILL.md) | 担当中の親同士の連絡、投稿する場面、受信時の判断と応答 | CLIの引数・状態・配送仕様（board README） |
| [.agents/memory/](memory/SUMMARY.md) | まだ規則でも仕様でもない再利用できる観察。共有5層と、全体調査が必要時に読む独立した [ops-report専用領域](memory/ops-report/SUMMARY.md)（境界はmemory skill） | コードやREADMEで確認できる事実、作業の進捗 |
| [.agents/ops/](#自動運用) | 自動運用（systemd --user timer、headless agent）と共通の決定的ツール | 対話作業の手順 |
| [knowledge/](../knowledge/README.md) | 実験結果・考察・関連論文（1 run = 1ノード） | 運用ルール |
| 各ディレクトリの `README.md` | モジュールの仕様・使い方・設計判断 | 横断的な運用ルール |
| GitHub issue / PR | 作業の目的・進捗・決定の経緯 | 恒久的な規則（決まったらここか AGENTS.md へ移す） |
| 各ツールのローカルmemory（Claude auto-memory など） | ユーザー個人の環境、そのツールの使い方に閉じた好み | プロジェクトの知見、ユーザーの判断（共有memoryへ） |

ツール固有のファイル:

- `CLAUDE.md`: `@AGENTS.md` を読み込むだけにする。`.claude/skills` は `.agents/skills` へのsymlink。
- `.codex/`: Codexの設定と参考資料。
- `.github/copilot-instructions.md`, `.github/agents/`: Copilot固有の承認フローだけを書く。

## 自動運用

自動運用は、kamimura ユーザーの systemd --user timer で動かす（self-hosted runner は隔離されているため使わない）。timer の有効化は、PRのmerge後に人間が行う。

| パス | 内容 |
|---|---|
| [ops/lib/agent_exec.sh](ops/lib/agent_exec.sh) | Claude（親、既定Opus）/ Codex（実装子、既定Sol・明示でAstra）を read-only または write で headless 起動・同一session継続・状態確認・停止する共通ランナー。既定値・選択値・実効値と終了理由をrun dirへ記録し、`task_success` は判定しない。`--profile` で job 別の固定値を選ぶ。契約は [ops/lib/README.md](ops/lib/README.md) |
| [ops/lib/agents.conf](ops/lib/agents.conf) | 自動運用で使う model・effort（Codexは service_tier も）の固定値。呼び出し環境のCLI設定で黙って変わらないよう、全自動運用がここを参照する。job 別のプロファイル（例: レビュー用の `REVIEW_CODEX_*`）もここに置く |
| [ops/lib/opsproc.py](ops/lib/opsproc.py) | ops job 共通のサブプロセス実行。agent はプロセスグループごと起動・停止し、子孫プロセスを残さない |
| [ops/install_units.sh](ops/install_units.sh) | `ops/*/systemd/` の unit を `~/.config/systemd/user` へリンクする。`--enable` を付けたときだけ timer を有効化する |
| [ops/ops-report/](ops/ops-report/README.md) | 親とnative scoutの並列全体調査、3日周期の運用レポート（未完了回は部分公開・再開）、採用提案の日次triage |
| [ops/review/](ops/review/README.md) | 親が明示起動する1回の統合Codexレビュー、指摘解消・coverageの記録、依存を照合する順次merge gate、共有基盤パスの一覧 |
| [ops/cleanup/](ops/cleanup/README.md) | worktreeを持たないmerge済みbranchの自動削除と、cleanなworktreeの承認制削除・候補一覧 |
| [ops/issues/](ops/issues/README.md) | 親子issueの書式の正本、生成・整合検査の共通APIと読み取りCLI |
| [ops/parent/](ops/parent/README.md) | 親セッションの担当・合意・子の進捗・明示レビュー要求を管理するCLI。会話からの手順は [parent-session](skills/parent-session/SKILL.md) |
| [ops/board/](ops/board/README.md) | 親issueに結び付く名前・名簿・投稿と状態・呼び鈴を管理する掲示板CLI |
| [ops/herdr_labels/](ops/herdr_labels/README.md) | 親担当をHerdrの表示へ同期する。会話からの設定・修復は [herdr-session-label](skills/herdr-session-label/SKILL.md) |

利用枠の不足などで自動運用が失敗したときは、黙ってskipせず記録する。

## GitHub運用ルール（初版）

- **issue起票**: AIは、作業中に見つけたバグ・負債・派生課題を自由に起票してよい。AIが起票したissueには `ai-proposed` ラベルを付ける。運用レポートで人間が整理する。
- **放置PR**: N日動きのないPRは、運用レポートに「継続 / rebaseして仕上げ / close」の選択肢付きで載せる。
- **親子issueの書式**: タイトル・ラベル・本文・正式なsub-issue関係・進捗更新と段階移行は [親子issue規約](ops/issues/README.md) に従う。手動起票・親からの分割・レポートtriageで同じ規約を使う。
- **親セッションの担当**: 人間が会話で親issueの番号・URLと担当意図を示したら [parent-session](skills/parent-session/SKILL.md) を使う。担当指定と要件合意を区別し、合意後に親が分割・委任を統括する。担当の排他・解除・引継ぎの正本は [parent CLIのローカルstate](ops/parent/README.md#責務と状態の正本)。
- **close**: 実装issueは、PRのmerge（`Closes #`）で自動closeする。研究テーマのissueは、収束後に人間がレポートを読んでcloseする。
- **merge**: 親が明示起動した統合Codexレビューと指摘解消を条件とし、merge前のCIの有無・完了・成功は条件にしない。blocker/majorの解消は親が修正差分・回帰テストで確認し、自動再レビューを行わない。親の明示mergeコマンドで各PRのcoverage・最新head/base/merge-base/diff・依存関係・競合を照合し、合格した独立PRから順次進める。CIは観測値として残し、merge後のmainのCIを確認する。既定はdry-run、人間の有効化判断後に呼出元環境の `AGENT_AUTO_MERGE=1` で実mergeする。共有基盤変更は人間の承認を要し、その親への明示合意を機械的に記録した場合だけ免除できる。記録・判定・停止/復旧の正本は [merge gate](ops/review/README.md#merge-gate)。運用レポート専用memoryの限定例外は [ops-report README](ops/ops-report/README.md#専用memoryの保存)。

## 研究ループ（概要）

- **単位**: 研究テーマが親issue、1サイクル（調査→選定→実験→考察）が子issue。実測した結果は `knowledge/` のノードに、次サイクルの提案は子issueの結論に書く。
- **人間が介入するのはテーマ設定だけ**: 問題提起・目標指標・評価条件・収束/停止条件を人間とAIが一緒に決め、テーマissueに固定する。以降の手法調査・選定・実験・次の調査は、AIが自律で回す。途中経過は運用レポートで観察する。
- **収束判定**: 共通のルールは持たない。テーマ設定時に、AIが指標の性質に応じた閾値・seed数・停止条件を提案し、合意して固定する。
- **ARIS**: そのまま導入しない。training-queue と knowledge に合わせた軽い自前skillとして作り直し、ARISからは有用な部品だけを取り込む。既定のskill群は肥大させない。
- 手順の正本: [skills/research-loop/SKILL.md](skills/research-loop/SKILL.md)

## このファイルの改訂

- ルールは仮説である。合わなくなったら、運用レポートの提案として改訂する。
- `AGENTS.md` と `.agents/**` は共有基盤なので、変更するPRは人間が承認する。
- 規則を追加する前に、memoryの観察で済まないか、既存の規則を消せないかを先に検討する。
