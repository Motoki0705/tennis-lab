---
name: parent-session
description: 人間が会話で親issueの番号・URLと担当意図を示すか、dispatcherのbootstrapが予約済みの親へ開発を指示したときに使う。実行根拠を照合し、子issue分割・Codex委任・進捗・明示レビュー要求とmergeを管理する。単なるissue参照や説明・要約の依頼では発動しない。
---

# 親issueを担当する

人間が「#Nを担当して」「この親issueを進めて」のように番号・URLと担当意図を示したときに使う。
「#Nを説明して」「このissueを参考にして」は割当ではないので登録しない。
人間に専用コマンドやセッションIDの入力を求めず、指定された親セッション自身がCLIを実行する。

共通運用は [.agents/README.md](../../README.md)、CLIの入出力と再試行・引継ぎ契約は
[parent README](../../ops/parent/README.md)、書式は [issues README](../../ops/issues/README.md) を参照する。
研究テーマ固有のサイクル設計はresearch-loopの範囲とし、このskillで停止基準を新設しない。
担当開始後の親同士の連絡・受信への応答は[parent-board](../parent-board/SKILL.md)に従う。

## 自律割当から担当する

dispatcherのbootstrapで起動された親は、予約と実runtimeを `accept` で結合してから
[開発workflow](../../ops/parent/development_workflow.md) に従う。下の手動担当のregister/checking/agreeを繰り返さない。
既存の自律根拠とissue/コメント・正式関係・依存・採用状態を照合し、通常の実装判断は親が具体化する。
人間合意や共有基盤許可を生成せず、重要な未決事項は理由・再開条件付きで待機する。
同じprivate stateでworkerとreview/mergeを扱う。Projectへの節目の接続予定も開発workflowを参照する。

## 担当と要件を確定する

1. 親issueの本文と全コメント、関連PR・既存の子を読み、最新の人間との合意を確認する。
   CLIの `status` で対応を確認し、`register` で担当登録する。別の担当や別の親が表示されたら
   黙って切り替えない。既存担当の解除・終了・明示引継ぎをREADMEに従って扱う。
   Herdrでは結果の`display_sync`を確認する。表示の設定・修復には
   [herdr-session-label](../herdr-session-label/SKILL.md)を使い、名前・担当番号・短い題名を表示する。
   `board_sync`と`board_delivery`の成否も確認し、未読・配送の問題はparent-boardの手順で扱う。
2. 採用済みのレポート提案は `receive-proposal` で要件確認の対象として明示受領する（契約はparent README）。
   受領の再送は冪等で、以前の親合意を新しい提案へ流用しない。人間作成issueも同じ登録入口から進める。
   親自体の初回合意・見直しでは `checking` を記録する。合意済みの親へ追加提案を受領した場合は
   親のphaseを戻さず、その提案の目的・範囲・期待する動作・受入条件を整理する。
   予備調査に基づく仮説と選択肢を添え、不明点を専用の質問機能で人間と認識合わせする。
   既に記録された合意は再確認せず、変更・不足がある論点だけを詰める。
   **担当指定や改善案の採用だけを実装への合意と見なさない。**
3. 人間と合意した内容と確認発言の引用を `agree` で親issueへ記録する。
   提案の合意には `--report` / `--proposal` で対象を明示する（契約はparent README）。
   未合意の提案由来の子作成・実装委任・実装を保留し、独立した合意済み作業は継続する。
   合意が揃ったら確定した方針と前提を伝えて着手する。
   合意済み範囲の分割・担当割当について追加承認を求めない。
   人間作成issueもレポート由来issueも同じ順序で扱う。単独で実装できるissueは不要な階層を増やさない。

合意後に親コメントの鮮度不一致で操作が止まったら、エラーのID・URLから追加・編集・削除を確認する。
要件と無関係な現存コメントだけ、[parent README](../../ops/parent/README.md#入口と合意) の
`agreement-ignore-comments` でdigest・理由を明示して分類する。人間の指示を無関係として除外しない。
要件変更・停止指示・削除は `checking` → 人間との認識合わせ → `agree` で再合意する。
旧合意の読取や進捗記録の継続と、変更後の委任・レビュー・mergeの許可を混同しない。

## 分割し、委任する

1. 実装単位・依存関係・担当を決め、issues APIを使う `create-child` で起票する。
   自由文の書式をコピーせず共通specを渡す。キーを固定し、再送時も同じキーで既存子を再利用する。
2. `status` / `reconcile` の `launch_plan` で完了済み・既存PR・既存委任と起動対象を照合する。
   **依存が許す子は、設定された枠内ですべて並列に起動するのが既定。** 手動親の既定枠は全子であり、
   固定3件にはしない。自律ループの各親の実装子2件という初期値は別の実行枠で、その設定を守る。
   各子の専用worktreeを作り、既定のCodexへ共通ランナー経由で委任する。
   自律親は開発workflowの `start-worker`（delegate → dispatcher start-child）、
   手動親は `delegate` の `action: launch` 後に [共通ランナー](../../ops/lib/README.md) を使い、この親セッション内で直ちに起動する。
   先に起動した子の完了を待たず、残る `launchable_children` を再計算してすべて委任する。
   子の番号・担当範囲・対象外・受入条件・依存先・worktree・検証条件と共有pytest lockを渡す。
   子が実装・検証・PRを作り、親が成果物と検証記録を集約する。workerの終了はrunnerの証拠と照合し、
   `record-worker-ended` で当該 `delegation_id` の予約枠を解放する。PR作成や停止記録だけから終了を推測しない。
3. PRに各子を閉じる指示を記載し、`record-result` で子の成果物欄にPRと検証結果を残す。
   `reconcile` と `sync` で子⇔担当/branch/worktree⇔PRと全体進捗を照合する。
   単独issueのworkerには自身の番号を使う。PR作成後の修正も自律親の `continue-worker` で同じsessionへ戻し、
   終了・差分・テスト・PRの確認を省略しない。dshの明示指定はAGENTS.mdの規定に従う。

### 並列にしない理由と資源の扱い

- 起動しない子は `launch_plan.tasks[].reasons` に理由が必要。計画にある依存・上限・既存PR・予約の理由は
  `sync` で親の進捗へ記録する。計画だけでは表せない依存・資源の例外で逐次にする子には、
  詳細と再開条件を `stop` に記録して `sync` する。単に未着手のままにしない。
  上限を変える場合は [configure-delegation](../../ops/parent/README.md#並列起動の選択と予約上限) へ実測と理由を渡す。
  親issueの進捗に起動予約の一覧・起動可能な子・待機理由が表示されることを確認する。
- 人間の判断待ちは**その子だけを `stop`** し、影響する依存子だけを保留する。
  起動時にも適用し、独立した子は同じターンで起動・継続する。
- pytestは [共有lockラッパー](../../ops/parent/README.md#pytest共有lock) で1件ずつ実行する。
  テスト待ちはworker全体を逐次にする理由にしない。GPU・Blenderは既存の
  [training queue](../training-queue/SKILL.md) の `half` / `all` 枠とメインrepoの共有stateを使う。

### 依存先がPR作成済みならstacked branchで進める

1. 依存先のPR・branch・headと停止事項を確認する。merge待ちだけで後続を止めず、
   `launch_plan.tasks[].stacked_on` の全headを含む専用branch/worktreeを作る。
   依存が直列なら最後のbranchへ積み、独立した複数の依存headなら専用の統合branchへ取り込む。
   競合や未確定の仕様を解消できなければ、その子に理由を記録し、独立した子を進める。
2. [stacked branchのCLI手順](../../ops/parent/README.md#stacked-branchの確認とrebase) に従い、
   起動時の依存headを固定して `delegate` する。CLIは全依存headが子のHEADの祖先であることを検査する。
   stacked branchのPR（`Closes #子`）は、先行PRのmerge後にrebaseしてから作る。親単位レビューとmerge gateは
   全open子PRの集合をレビュー時の対象と照合するため、先行PRのレビュー後に後続のPRを開くと
   先行PRのgateが `pr_set_changed` で止まる。それまではbranchのpushと子issueへの記録にとどめる。
   先行branchが更新されたら差分・インターフェースを確認して積み直す。
3. 先行PRのmerge後は、固定した古い依存headを境界としてmainへ `rebase --onto` する。
   squash mergeでも先行の変更を重複させず、子自身の差分・テストを確認しPRのbaseをmainへ変更する。
   レビュー対象のhead/base/diffが変わるため、親の対象固定はrebase後に行う。
   既存レビューのcoverageを変更後へ流用せず、必要な新要求は理由付きで明示する。自動再レビューはしない。

## 進捗と部分停止を扱う

- 親issueへ自由文の進捗・連絡を書くときは `note --body-file` を使い、素のコメントを書かない。
  `sync` 等の既存マーカー付き記録はそのまま使う。noteは要件変更の記録にはならず、変更は `checking` → `agree` で扱う。
- 子の停止時は `stop` で理由・確認済みの事実・依存作業への影響・再開条件・次の対応を残す。
  詳細は子、親にはリンク付きの短い通知と全体進捗を置く。影響する依存作業だけを保留し、独立した子は継続する。
- 条件が満たされた証拠を記録して `resume` し、保留を再計算する。
  要件変更が必要ならその変更について人間と合意してから再開する。
- 今回必須ではない発見は `improvement` で子の次の改善候補へ目的・根拠・想定範囲を残す。
  候補の発見だけで受入条件・実装範囲を広げず、現在の実装を続ける。

## レビュー要求と完了確認

親が準備できたと判断した時点で、その時点の対象PR群を確定し、`review-prepare` で
head/base/merge-base/diff SHA-256とreview run IDを固定する。
PR群・受入条件・統合上の確認点を明示して `review-request` を実行し、READMEのJSON契約で
[親単位レビュー入口](../../ops/review/README.md) の `parent_review.py run --request` へ引き渡す。
担当親自身が同じ環境・stateで実行する。PR作成・push・CI完了はレビュー指示にしない。

結果を受領したら状態と証拠を記録し、阻害指摘を担当子へ戻す。修正差分・CI・回帰テスト・
指摘解消を親が確認し、`parent_review.py resolve` で指摘ごとの修正head・実差分・CI・回帰テスト・
解消根拠を記録する。coverage不成立・失敗・部分結果を成功にしない。自動再レビューを起動しない。
障害後の再送でもCodexは再実行しない。追加のレビューが必要なら理由と新しいrun IDの明示指示を残す。
結果を受領した証拠を `review-state --state result_received`、修正差分・CI・回帰テストと指摘解消の確認を
`review-state --state fixes_verified` へ記録する。指摘なしでも親の確認を記録する。
続けて [merge gate](../../ops/review/README.md#merge-gate) の `merge_gate.py --parent N merge` を明示実行する。
既定のdry-runで理由と実効設定を読み、人間が有効化を合意済みならREADMEの環境変数で実mergeする。
共有基盤を含めた親mergeの合意が既にある場合は、同READMEの `shared-permission` で根拠を記録する。
条件を満たすPRは依存順に進み、親全体の完了まで独立PRをためない。
merge後の子の自動closeと親進捗への反映を確認する。停止・未完了や試行済み記録が出たら、
実状態と保存証拠を照合して復旧し、gateを通すための自動再レビューや同headへの二重mergeを行わない。

全体の受入条件、未解決の停止・人間判断、子の検証・PRの完了を照合する。
チェック数だけを完了根拠にしない。全体が完了したら親へ証拠と残した改善候補を記録し、
手動親は `release --ended` で担当を終了する。自律親は開発workflowの `workflow-report` で実merge/closeと証拠を照合し、
4キーの最終イベントを返す。supervisorが子の終了と最終イベントを確認してから担当を終了する。

## 中断・引継ぎから再開する

同じセッションは `status` と `reconcile`、別セッションは明示的な `handoff --accept` から始める。
Herdrの表示はreconcile/handoffに連動して復元・解除される。表示だけ失敗した場合は
herdr-session-labelで再同期し、担当操作を成功扱いのまま隠さない。
issue/PRの実状態と保存された委任・レビューrunを照合し、完了済みの再実装と二重起動をしない。
既存worker・PR・worktreeのある子はそこへ戻す。起動予約後に失敗した子は旧workerの終了を確認してから、
READMEの証拠付き再予約を使う。未指定・終了済み・引継ぎ待ちを成功や自動代替と扱わない。

引継ぎ前に、親へ全体の進捗・対象PR/run・次の作業、子へ停止の詳細と検証記録を残す。
ローカルのセッション識別子やworktreeの絶対パスをissue・コメント・memoryへ貼らない。
