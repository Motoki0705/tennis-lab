---
name: parent-session
description: 人間が親issueの担当や作業（実装・残作業・調査の続き・引継ぎ）を指示するか、dispatcherのbootstrapが予約済みの親へ開発を指示したときに使う。子issueを作らない直接実装も含む。実行根拠を照合し、担当登録・要件合意・子issue分割・Codex委任・報告の検証・進捗・明示レビュー要求とmerge・中断からの再開を管理する。説明・要約・参照だけの依頼では発動しない。
---

# 親issueを担当する

人間が「#Nを担当して」「この親issueの残作業を進めて」「#Nを引き継いで」のように、番号・URLと担当の意図を示したときに使う。
子issueを作らず直接実装する依頼も担当として扱い、[担当と要件を確定する](#担当と要件を確定する)から始める。
「#Nを説明して」「このissueを参考にして」は割当ではないので登録しない。
人間に専用コマンドやセッションIDの入力を求めず、指定された親セッション自身がCLIを実行する。

共通運用は [.agents/README.md](../../README.md)、CLIの入出力と再試行・引継ぎ契約は
[parent README](../../ops/parent/README.md)、書式は [issues README](../../ops/issues/README.md) を参照する。
研究テーマ固有のサイクル設計はresearch-loopの範囲とし、このskillで停止基準を新設しない。
担当開始後の親同士の連絡・受信への応答は[parent-board](../parent-board/SKILL.md)に従う。

## 流れ

1. [始める前に](#始める前に): 最新の規則を読み、parent CLIを動かすcheckoutを決める。
2. [担当と要件を確定する](#担当と要件を確定する): `register` → `checking` → 人間と認識合わせ → `agree`。
3. [分割し、委任する](#分割し委任する): `create-child` → `launch_plan` → 子ごとのworktree → worker起動。
4. [子の報告を受け取る](#子の報告を受け取る): 終了の照合 → 差分・テスト・PRの確認 → `record-result`。
5. [レビューを要求する](#レビューを要求する): CI完了後に `review-request` → 結果受領 → 修正確認。
6. [mergeと完了確認](#mergeと完了確認): merge gate → mainのCI → 全体の受入確認 → 担当終了。

親issueへの書き込みは[記録の経路](#親issueへ記録する)に従う。CLIが止まったら[対処表](#止まったときの対処)を引く。

## 自律割当から担当する

dispatcherのbootstrapで起動された親は、予約と実runtimeを `accept` で結合してから
[開発workflow](../../ops/parent/development_workflow.md) に従う。下の手動担当のregister/checking/agreeを繰り返さない。
既存の自律根拠とissue/コメント・正式関係・依存・採用状態を照合し、通常の実装判断は親が具体化する。
人間合意を生成せず、重要な未決事項は理由・再開条件付きで待機する。
同じprivate stateでworkerとreview/mergeを扱う。Projectへの節目の反映も開発workflowを参照する。
[始める前に](#始める前に)と[親issueへの記録](#親issueへ記録する)は自律親にも適用する。

## 始める前に

- `git fetch origin` し、このskill・AGENTS.md・共有memoryを `origin/main` の版で読む
  （`git show origin/main:<path>` か、`origin/main` から作ったworktree）。メインtreeのローカル `main` は自動で更新されず、
  古い規則が残る。古い規則を子への指示へ写すと、子が誤った運用をする。
- parent CLIを動かすcheckoutを決め、担当中は変えない。同じ委任のworkerの起動と継続は同じcheckoutから行う。
  共通ランナーは継続時に設定ファイル（`agents.conf`）のパスと内容も照合するため、別のcheckoutからの継続も、
  その間に設定ファイルが変わった継続も拒否される。checkoutの切替や更新は、継続が残る委任のない区切りで行う。
- `status` で現在の担当・合意・委任を確かめる。中断した担当の再開なら[中断・引継ぎから再開する](#中断引継ぎから再開する)へ進む。

## 担当と要件を確定する

子issueを作らず直接実装する場合も、以下の手順を省略しない。
自分で実装するか、未分割の単独issueとしてCodexへ委任するかは、人間の指示と既存の手順に従う。

1. 親issueの本文と全コメント、関連PR・既存の子を読み、最新の人間との合意を確認する。
   `register` で担当登録する。別の担当や別の親が表示されたら黙って切り替えず、
   既存担当の解除・終了・明示引継ぎを[parent README](../../ops/parent/README.md#担当の終了引継ぎ)に従って扱う。
2. 書式を整える。タイトル先頭に `[識別名]` がなければ、内容から2〜6文字の識別名を決め、
   `gh issue edit <親番号> --title "[識別名] 元の要約"` で改名する。
   `issue:parent` がなければ付与し、本文の親の必須節に不足があれば既存内容を保持して骨組みを補完する。
   識別名・タイトル・必須節の規則は [issues README](../../ops/issues/README.md) に従う。
   本文の補完は合意の前に済ませる（合意後の本文の変更は[親issueへ記録する](#親issueへ記録する)の制約を受ける）。
3. 表示を確かめる。掲示板の名前とHerdr表示は `register` で付く。結果の `display_sync` を確認し、表示の設定・修復には
   [herdr-session-label](../herdr-session-label/SKILL.md)を使い、名前・担当番号・短い題名を表示する。
   `board_sync` と `board_delivery` の成否も確認し、未読・配送の問題はparent-boardの手順で扱う。
4. 要件を確認する。親自体の初回合意・見直しでは `checking` を記録する。
   採用済みのレポート提案は `receive-proposal` で要件確認の対象として明示受領する（契約はparent README）。
   受領の再送は冪等で、以前の親合意を新しい提案へ流用しない。人間作成issueも同じ登録入口から進める。
   合意済みの親へ追加提案を受領した場合は親のphaseを戻さず、その提案の目的・範囲・期待する動作・受入条件を整理する。
   予備調査に基づく仮説と選択肢を添え、不明点を専用の質問機能で人間と認識合わせする。
   既に記録された合意は再確認せず、変更・不足がある論点だけを詰める。
   mergeの可否と時期（レビューとCIの後にmergeする、人間が結果を見るまでmergeしない等）も要件として確かめる。
   **担当指定や改善案の採用だけを実装への合意と見なさない。**
5. 人間と合意した内容と確認発言の引用を `agree` で親issueへ記録する。
   提案の合意には `--report` / `--proposal` で対象を明示する（契約はparent README）。
   未合意の提案由来の子作成・実装委任・実装を保留し、独立した合意済み作業は継続する。
   合意が揃ったら確定した方針と前提を伝えて着手する。
   合意済み範囲の分割・担当割当について追加承認を求めない。
   人間作成issueもレポート由来issueも同じ順序で扱う。単独で実装できるissueは不要な階層を増やさない。

`checking` / `agree` が書き直しの確定待ちで拒否されたら、[wait-rewrite](../../ops/parent/README.md#書き直しの確定待ち)で確定を待つ。
マーカーのない人間との合意しかない既存issueや、本文digestのない旧合意で止まった親は、
[稼働中の親への移行](../../ops/parent/README.md#稼働中の親への移行)に従って同じ内容を記録する。

## 親issueへ記録する

人間とagentは同じGitHub loginを使うため、親issueの素のコメントは人間の指示と区別できない。
合意後に素のコメントが増えると、委任・レビュー・merge前の鮮度検査で操作が止まる。記録の経路は内容で選ぶ。

| 記録すること | 使う入口 |
|---|---|
| 進捗・連絡・調査結果・人間への報告 | `note --body-file`（[自由文の進捗・連絡](../../ops/parent/README.md#自由文の進捗連絡)） |
| 子の一覧と全体進捗 | `sync`（親本文の進捗節を更新する） |
| 子の停止・再開・改善候補 | `stop` / `resume` / `improvement` |
| 人間の判断で目的・範囲・期待する動作・受入条件（mergeの扱いを含む）が変わる | `checking` → 人間と認識合わせ → `agree`。noteや素のコメントで代えない |
| レビューの対象・指示・状態 | `review-request` / `review-state` |

- 合意後に親本文を手で直すのは、進捗の4節（子タスクのチェックリスト・進捗概要・成果物・関連PR・次の改善候補）と、
  完了条件のチェックの付け外しだけにする。他の節は合意の本文digestに入り、変えると委任・レビュー・mergeが
  `human agreement requirements changed` で止まる。要件を変えるときだけ再合意と合わせて書き換える。
  誤って書き換えたら、再合意の前に合意時点の本文へ戻す。digestが戻れば、止まった操作はそのまま通る。
- 子workerには親issueへコメントさせない（[workerへの指示](#workerへの指示)）。
- 鮮度不一致（`unclassified agreement comments`）で止まったら、エラーのID・URLから追加・編集・削除を確認する。
  要件と無関係な現存コメント（進捗、workerの報告など）だけを、[parent README](../../ops/parent/README.md#合意進捗コメント)の
  `agreement-ignore-comments` でdigest・理由を明示して分類する。人間の指示を無関係として除外しない。
  要件変更・停止指示・削除は `checking` → 人間との認識合わせ → `agree` で再合意する。
  旧合意の読取や進捗記録の継続と、変更後の委任・レビュー・mergeの許可を混同しない。

## 分割し、委任する

1. 実装単位・依存関係・担当を決め、issues APIを使う `create-child` で起票する。
   自由文の書式をコピーせず共通specを渡す。キーを固定し、再送時も同じキーで既存子を再利用する。
   `spec.labels` に作業種別に加えて対象領域ラベルを1〜2件明示し、`ai-proposed` は付けない。
   領域は `module: base` / `module: ball-detection` / `module: court-detection` / `module: blcs` /
   `module: plcs` / `module: submodules` / `module: synthetic-data` / `module: tennis-scene` /
   `ci` / `tests` / `dependencies` / `third-party` / `knowledge` 等から選ぶ。
   一覧の正本は [.github/labels.yml](../../../.github/labels.yml) の `area` 軸を参照する。
   テストの設計は[テスト設計を委任へ渡す](#テスト設計を委任へ渡す)に従って子の本文へ書く。
   子issueの「依存関係」節の `#番号` はPR番号もすべて先行依存として解析されるため、参照用のPR番号は別の節に書く。
   解析仕様は [子manifestの説明](../../ops/parent/README.md#子の作成委任再開) を参照する。
2. `status` / `reconcile` の `launch_plan` で完了済み・既存PR・既存委任と起動対象を照合する。
   **依存が許す子は、設定された枠内ですべて並列に起動するのが既定。** 手動親の既定枠は全子であり、
   固定3件にはしない。自律ループの各親の実装子2件という初期値は別の実行枠で、その設定を守る。
3. 起動対象ごとに、メインcheckoutの `.claude/worktrees/` 配下へ専用のbranch・worktreeを作り、
   既定のCodexへ共通ランナー経由で委任する。
   自律親は開発workflowの `start-worker`（delegate → dispatcher start-child）、
   手動親は `delegate` の `action: launch` 後に [run-worker](../../ops/parent/README.md#手動workerの起動継続とrun記録) をこの親セッションからバックグラウンド実行する。
   この入口だけを使い、Codex CLIは直接実行しない。モデルは [AGENTS.md](../../../AGENTS.md#開発ルール) の判断基準に従い、
   通常はSol/max、高難度の数理・幾何アルゴリズムや広範囲の統合設計では親がAstra/maxを選べる。
   人間との合意で指定されたモデルを優先する。選択理由を親issueの `note` と具体指示へ書き、
   手動親は `run-worker --model-reason-file` でも委任・runへ保存する（既定採用の理由は入口が記録する）。
   初回だけ必要に応じて `--model` を渡し、effortはmax固定で指定しない。
   resumeの省略は初回モデルを継承し、同値の再指定は許可、異値は拒否される。同一委任・sessionの途中で変えない。
   利用不可や実効値不一致を別モデル・別sessionで補わず、理由と再開条件を親へ返す。
   dshを案件で明示指定された場合は [deepseek-harness skill](../deepseek-harness/SKILL.md) に従う。
4. 先に起動した子の完了を待たず、残る `launchable_children` を再計算してすべて委任する。
5. 子が実装・検証・PRを作り、親が成果物と検証記録を集約する（[子の報告を受け取る](#子の報告を受け取る)）。
   PRには各子を閉じる指示（`Closes #子`）を書かせる。単独issueのworkerには自身の番号を使う。
   `reconcile` と `sync` で子⇔担当/branch/worktree⇔PRと全体進捗を照合する。

### workerへの指示

`run-worker` と `start-worker`（どちらも継続を含む）は、[標準のworker指示](../../ops/parent/worker_prompt.md)、
親・担当issueの番号、worktree・branch、担当issueの原文を先頭に置き、指示ファイルの内容を「親の具体的な指示」として続けて渡す。
指示ファイルには案件固有のことだけを書く。

- 子の番号・担当範囲・対象外・受入条件・依存先と固定したhead・PRを作る時点
- テストの設計（下の節）と検証条件
- 親が決めた判断と前提、使ってよい資源（GPU・Colab・Drive等）とその予算
- 継続のターンでは、残る指摘・今回の修正範囲・確認済みの事実

CI・コミット・PRの書式などrepo共通の規則は写さない。標準指示とAGENTS.mdにあり、古い版を写すと子が誤る。
workerは報告・質問を最終メッセージで返し、親issueへはコメントしない（標準指示に含む）。

### テスト設計を委任へ渡す

親は [tests/README.md のテスト作成方針](../../../tests/README.md#テスト作成方針)に沿って設計し、
子issueの「検証方法・証拠」の行に、再利用する既存テストの node・新しく置く層・検出したい不具合を具体的に書く。
設計の5点をその行またはそこから参照する作業計画にまとめ、委任時に同じ設計を子へ渡す。
受入条件1つにつき新規テスト1つという機械的な割当はしない。

設計前に調査が必要なら、子の最初の作業を既存テストの調査と設計案の提案にし、親が承認してから実装へ進める。
子から設計変更の提案が来たら、必要性とコストを確認して子issueの設計を更新し、指示を返す。

子issueの記入例（既存 unit の拡張で契約を守る場合）:

```markdown
- [ ] 不正batchをモデル実行前に拒否する
  文脈: 無効な入力をモデル計算へ流さないため。
  下流への影響: adapter経由の推論・学習で、入力破損を早期に検出する。
  検証方法・証拠: 契約は不正batchの計算前拒否、検出したい不具合は検証より先にforwardする変更。既存nodeは tests/unit/tasks/base/model_io/test_contracts.py::test_invalid_batch_fails_before_model_forward。新しく置く層はunit（既存nodeを拡張し、新規e2eなし）。コスト上限は追加1ケースでREADMEの時間基準内、--collect-onlyと--durations=0で確認する。
```

レビュー依頼には、渡した設計への参照、設計からのずれと理由、テストの増分（追加・削除・統合した node、展開後の件数・実測時間）を含める。
新規テストがないことだけで差し戻さず、既存テストによる証拠と未検証の契約を確認する。

### 並列にしない理由と資源の扱い

- 起動しない子は `launch_plan.tasks[].reasons` に理由が必要。計画にある依存・上限・既存PR・予約の理由は
  `sync` で親の進捗へ記録する。計画だけでは表せない依存・資源の例外で逐次にする子には、
  詳細と再開条件を `stop` に記録して `sync` する。単に未着手のままにしない。
  上限を変える場合は [configure-delegation](../../ops/parent/README.md#並列起動の選択と予約上限) へ実測と理由を渡す。
  親issueの進捗に起動予約の一覧・起動可能な子・待機理由が表示されることを確認する。
- pytestは [並列実行の手順](../../ops/parent/README.md#pytestの並列実行) に従い、親・子・worktree間で並列実行してよい。共有lockは使わない。
  テスト待ちはworker全体を逐次にする理由にしない。GPU・Blenderは既存の
  [training queue](../training-queue/SKILL.md) の `half` / `all` 枠とメインrepoの共有stateを使う。

### 依存先がPR作成済みならstacked branchで進める

依存先が親CLI・共通ランナー・worker指示など委任の基盤自体を変える場合は、例外として依存先のmerge後にmainから開始する。
旧CLIで委任すると、merge後に `migrate-worker-phase` などの[明示移行](../../ops/parent/README.md#workerの調査親判断実装フェーズ)が必要になるため。

1. 依存先のPR・branch・headと停止事項を確認する。merge待ちだけで後続を止めず、
   `launch_plan.tasks[].stacked_on` の全headを含む専用branch/worktreeを作る。
   依存が直列なら最後のbranchへ積み、独立した複数の依存headなら専用の統合branchへ取り込む。
   競合や未確定の仕様を解消できなければ、その子に理由を記録し、独立した子を進める。
2. [stacked branchのCLI手順](../../ops/parent/README.md#stacked-branchの確認とrebase) に従い、
   起動時の依存headを固定して `delegate` する。CLIは全依存headが子のHEADの祖先であることを検査する。
   stacked branchのPR（`Closes #子`）は先行branchをbaseとして早期に作成・レビューできる。
   先行branchが更新されたら差分・インターフェースを確認して積み直す。
3. 先行PRのmerge後は、固定した古い依存headを境界としてmainへ `rebase --onto` し、
   子自身の差分・テストを確認してPRのbaseをmainへ変更する。
   [reviewの継承条件](../../ops/review/README.md#base取込みrebase後の継承) を満たせば再レビューせず、
   coverageと解消確認を引き継ぐ。それ以外の追加変更は親がresolveする。mergeは依存順に行う。

## 子の報告を受け取る

指示・回答・最終報告の保存結果も [workerログ保存](../../ops/worker_logs/README.md#状態と復旧cli) で確認する。
公開失敗は同じturnの保存だけを再開し、workerを起動し直したり実装PRへログを混ぜたりしない。

workerは既定でplanningから始める。最初の報告は [標準worker指示](../../ops/parent/worker_prompt.md) の6節で読み、
`decide-worker-plan --child N --action approve|investigate|reject --answer-file tmp/answer.md` で判断を記録する。
質問なしでも省略せず、既存の実行根拠の範囲内なら親AIが判断する。範囲・要件を変える判断だけ再合意へ戻す。
判断後の `run-worker --resume` / `continue-worker` は回答を自動配送するため `--prompt-file` を渡さない。
追加指示は `--additional-instruction-file` に分ける。planningの終了を成果登録へ進めず、実装ターンの終了後に以下を行う。
自明な初回作業だけ `--skip-planning-reason-file` で理由を明示して省略できる。
状態・失効・移行のCLI契約は [parent README](../../ops/parent/README.md#workerの調査親判断実装フェーズ) を参照する。

1. 終了を照合する。runnerのexit 0や `result=completed` は1ターンの終了であり、タスクの成功ではない。
   手動親は、`run-worker` の出力の `delegation_id` に対応する親stateの最新run（`worker-runs/<delegation_id>/` 配下）を
   `agent_exec.sh status --run-dir` で確かめ、最終報告をそのrunの `final.md` で読む
   （記録の一覧は[共通ランナーREADME](../../ops/lib/README.md#記録と結果jsonschema_version-1)）。
   自律親は `inspect-worker` を使う。全子の実行状況は [worker_status](../../ops/worker_status/README.md) で一覧できる。
2. 手動親は、終了と子孫の不在をrunnerの証拠と照合してから `record-worker-ended` で当該 `delegation_id` の予約枠を解放する
   （自律親はdispatcherが記録する）。PR作成や停止記録だけから終了を推測しない。
3. 子の最終文の転記だけで合格にしない。実diff、受入条件ごとの証拠、テストコマンドと結果、
   PRのclosing指示・branch・HEADを親が確認し、`record-result` で子の成果物欄にPRと検証結果を残す
   （すぐにレビューを要求するなら `review-request --prs` の証拠で兼ねてよい）。
4. 子からの質問・停止報告・修正の依頼には、同じsessionへの継続で答える。手動親は `run-worker --resume`、
   自律親は `continue-worker` を使う。実行中のworkerは親issueを読み直さないので、noteだけで回答しない。
   PR作成後の修正も同じsessionへ戻し、終了・差分・テスト・PRの確認を省略しない。

## 進捗と部分停止を扱う

- 子の停止時は `stop` で理由・確認済みの事実・依存作業への影響・再開条件・次の対応を残す。
  詳細は子、親にはリンク付きの短い通知と全体進捗を置く。影響する依存作業だけを保留し、独立した子は継続する。
- 人間の判断待ちは**その子だけを `stop`** し、影響する依存子だけを保留する。
  起動時にも適用し、独立した子は同じターンで起動・継続する。
- 条件が満たされた証拠を記録して `resume` し、保留を再計算する。
  要件変更が必要ならその変更について人間と合意してから再開する。
- 今回必須ではない発見は `improvement` で子の次の改善候補へ目的・根拠・想定範囲を残す。
  候補の発見だけで受入条件・実装範囲を広げず、現在の実装を続ける。

## 要件・根拠が変わったとき

再合意（`checking` → `agree`）では内容が同じでも新しいrevisionの根拠になり、引継ぎでは担当の世代が変わる。
どちらも実行根拠（authorization）が変わり、それより前の委任・結果記録・レビュー確認は新しい根拠で使えない。
子の要件本文を変えた場合も、その子の委任・結果・レビューが同じ扱いになる。

- 可能なら、実行中のworkerが区切りに達してから再合意する。実行中のworkerは旧要件のまま動き続ける。
- PRのある子は、先に同じsessionへ変更内容を渡して再調査し、親判断・実装ターンを経て、
  現在の要件でPRの差分と検証を確かめて `record-result` を記録する。
- PRのない子も、元runの終了・回収を確認し、同じsessionへ変更内容を渡して再調査する。
  手動は通常の `run-worker --resume`、自律は根拠・担当変更の再結合が必要なら `refresh-worker-plan` 後に `continue-worker` を使う。
  別sessionへの再委任で失効を迂回しない。
- 旧根拠・旧世代のレビュー確認ではmergeできない。対象PRを現在の要件で確かめ、新しい理由とrunでレビューを要求し直す。
  予定した引継ぎなら、修正確認まで済んだPRは引継ぎの前にmergeしておく。

## レビューを要求する

1. 親が差分・検証を確かめた、準備できたopen子PRを選ぶ。同時に準備できたPRは1回のrunへまとめてよい。
2. 対象HEADのCI（`Python` を含む全check run）の完了を待ってから要求する。CIが実行中だとreviewerは確認できず、
   PRを `not_reviewed`（verdict `incomplete`）にし、通常1回の予算を超える追加runが必要になる。CIが失敗したら修正を先にする。
   PR作成・push・CI完了はレビューを自動で起動する契機ではない。レビューは親の明示指示で始める。
3. `review-request --prs` に成果の証拠と受入条件・統合上の確認点を渡す。指示には次も書く。
   - レビューの目的と判断の対象外（例: 方式の採否は人間が比較を見て決める）
   - テスト設計への参照・ずれ・増分（[テスト設計を委任へ渡す](#テスト設計を委任へ渡す)）
   - レビュー環境で取得できない証拠（CIログ本文、隔離CPUテスト等）があればその事実と、代わりに使える証拠
     （固定HEADのcheck runの結論等）。取得できないことだけを理由に `not_reviewed` にせず、
     それなしでは判断できない点がある場合だけ具体的に挙げるよう求める。
4. `review-request` は終了記録の照合・成果登録・対象固定・明示指示をまとめる。
   READMEのJSON契約で [親単位レビュー入口](../../ops/review/README.md) の `parent_review.py run --request` へ引き渡す。
   担当親自身が同じ環境・stateで実行する。
5. 結果を受領した証拠を `review-state --state result_received` へ記録し、阻害指摘を担当子へ戻す。
   修正差分・CI・回帰テスト・指摘解消を親が確認し、`parent_review.py resolve` で指摘ごとの修正head・実差分・CI・回帰テスト・
   解消根拠を記録してから、`review-state --state fixes_verified` を記録する。指摘なしでも親の確認を記録する。
6. coverage不成立・失敗・部分結果を成功にしない。自動再レビューを起動しない。
   起動回数0の準備失敗だけは同run IDで明示再試行できる。起動後の障害では再実行しない。
   追加のレビューが必要なら、理由と新しいrun IDの明示指示を残す。

## mergeと完了確認

1. [merge gate](../../ops/review/README.md#merge-gate) の `merge_gate.py --parent N merge` を明示実行する。
   既定のdry-runで理由と実効設定を読み、人間が有効化を合意済みならREADMEの環境変数で実mergeする。
   条件を満たすPRは依存順に進み、親全体の完了まで独立PRをためない。
2. PRへのpushごとにCIを実行し、コミットに `[skip ci]` を付けない。対象HEADの `Python` check runの成功を
   mergeの必須条件とし、未完了・失敗・取消・欠落・別HEADなら停止する。
3. PRのCIは最後のpush時点のmainとの組み合わせの結果である。その後に同じモジュール・関数・設定を変えたPRが
   mainへ入っていたら、mainを取り込んでCIを通し直してからmergeする
   （[base取込み後の継承](../../ops/review/README.md#base取込みrebase後の継承)を満たせば再レビューは不要）。
4. merge後の子の自動close、親進捗への反映、mainのCIを確認する。mainのCIが失敗したら、その修正を最優先で扱う。
   停止・未完了や試行済み記録が出たら、実状態と保存証拠を照合して復旧し、
   gateを通すための自動再レビューや同headへの二重mergeを行わない。
5. 全体の受入条件、未解決の停止・人間判断、子の検証・PRの完了を照合する。チェック数だけを完了根拠にしない。
   全体が完了したら親へ証拠と残した改善候補を記録し、手動親は `release --ended` で担当を終了する。
   自律親は開発workflowの `workflow-report` で実merge/closeと証拠を照合し、4キーの最終イベントを返す。
   supervisorが子の終了と最終イベントを確認してから担当を終了する。

## 中断・引継ぎから再開する

- 同じセッションは `status` と `reconcile`、別セッションは明示的な `handoff --accept` から始める。
  旧セッションが未解除のまま失われた場合の復旧と証拠は[parent README](../../ops/parent/README.md#担当の終了引継ぎ)に従う。
- Herdrの表示はreconcile/handoffに連動して復元・解除される。表示だけ失敗した場合は
  herdr-session-labelで再同期し、担当操作を成功扱いのまま隠さない。
- issue/PRの実状態と保存された委任・レビューrunを照合し、完了済みの再実装と二重起動をしない。
  既存worker・PR・worktreeのある子はそこへ戻す。起動予約後に失敗した子は旧workerの終了を確認してから、
  READMEの証拠付き再予約を使う。未指定・終了済み・引継ぎ待ちを成功や自動代替と扱わない。
- 引継ぎで世代が変わった後の既存の委任・結果・レビューは、[要件・根拠が変わったとき](#要件根拠が変わったとき)と同じ扱いになる。
- ホスト・WSLの再起動や会話の異常終了でworkerが記録なしに消えたら、`agent_exec.sh status --run-dir` で状態を見る。
  `running` の残存や `cleanup_failed` を成功・稼働中と扱わない。
  [共通ランナーの復旧手順](../../ops/lib/README.md#状態確認停止中断からの復旧)で全子孫の回収を確かめて
  `stop --cleanup-confirmation` を記録するまで予約を消さず、同じ作業のworkerを重複起動しない。
- 引継ぎ前に、親へ全体の進捗・対象PR/run・次の作業、子へ停止の詳細と検証記録を残す。
- ローカルのセッション識別子やworktreeの絶対パスをissue・コメント・memoryへ貼らない。

## 止まったときの対処

| 表示・状況 | 主な原因 | 対処 |
|---|---|---|
| `issue rewrite is pending` | 親issueの書き直しが未確定 | `wait-rewrite` で確定を待つ |
| `unclassified agreement comments` | 合意後に親issueへ通常コメントが追加・編集・削除された | [親issueへ記録する](#親issueへ記録する)の分類または再合意 |
| `human agreement requirements changed` | 合意後に親本文の進捗節以外が変わった | 誤った編集なら合意時点の本文へ戻す。要件の変更なら再合意 |
| `legacy human agreement lacks requirements_sha256` | 本文digestのない旧合意 | [稼働中の親への移行](../../ops/parent/README.md#稼働中の親への移行) |
| `worker delegation inputs changed` / `execution basis changed` | 再合意・引継ぎ・子の要件変更で根拠が変わった | [要件・根拠が変わったとき](#要件根拠が変わったとき) |
| `existing PR: reconcile instead of starting another worker` | PRのある子へ新しい委任を作ろうとした | 既存PRの担当sessionへ `run-worker --resume` で戻す（根拠が変わった後なら先に `record-result`） |
| `worker already has a run; inspect it and use --resume` | 同じ委任で初回起動を繰り返した | 前回runを確かめて `--resume` |
| `ended delegation requires an explicit resume with restart evidence` | 終了記録済みの委任を起動しようとした | `run-worker --resume --restart-evidence-file` |
| `worker termination and cleanup must be confirmed before resume` | 前回runの終了・回収が未確認 | `agent_exec.sh status`。記録なしに消えたrunは[共通ランナーの復旧手順](../../ops/lib/README.md#状態確認停止中断からの復旧) |
| `implementation reservation limit reached` | 予約上限に達している | 終了を確かめた委任を `record-worker-ended`。上限の変更は実測と理由を添えて `configure-delegation` |
| `manual worker is blocked: …` / `cannot be delegated: …` | 停止・依存・既存PR・予約などの計画上の理由 | 表示された理由を解消する。起動のためだけに停止を解除したり再委任したりしない |
| `run-worker` がexit 2・`runner_summary: null` で終わり、以後の継続が `runner records unavailable …` で止まる | 共通ランナーが起動前に拒否した（別のcheckoutからの継続、設定ファイルの変更など）。存在しないrunが記録に残る | 起動したcheckoutを確かめ、[#1610](https://github.com/Motoki0705/tennis-lab/issues/1610)の回復手順を使う |
| レビューが `incomplete`（PRが `not_reviewed`） | CIが未完了、またはCIログ・隔離CPUテストを取得できない | CI完了後、取得できない証拠と代替を指示に書き、新しい理由とrunで要求する |
