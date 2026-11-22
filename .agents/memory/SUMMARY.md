# 共有memory SUMMARY

各層の要点はエントリから自動生成する。作業を始める前に読み、詳細は層の索引から辿る。[memory skill](../skills/memory/SKILL.md) に読み方と更新手順がある。memoryの観察とコードが食い違った場合はコードを確認する。

## [workflow](workflow/index.md) — 作業のコツ

- 親たちへの一斉指示は、応答した親から進め、人間の操作が必要な手順を送信した親がまとめて伝える。

## [decision](decision/index.md) — ユーザーの判断

- 判断を仰ぐのは未決の事項だけにし、各選択肢に具体的な説明を添える。
- 2026-06-21のPLCSローカル整理では、指標記録済みでbackfill不能な旧ckptの手動削除を人間が認めた。通常のpruneは検証済み予測を要求する。
- Colabの複数条件の学習は1セッションで順に実行し、VMを作り直しても保存済みの段階とcheckpointから再開できるようにする。
- 独立した目的の変更は別PRに分ける。同じ目的を成立させる実装・テスト・文書はまとめる。
- 一度限りの実験進行用コードは常設のsrcから除き、再現に必要なscriptと実行時点の記録をknowledgeに残す。
- 公開文書は英語を既定にし、日本語・簡体字中国語を別ファイルで維持する。資産のハッシュ値はコードに置く。
- 実装名から機能・方式が伝わるようにし、Issue番号は追跡情報に置く。knowledgeの既存run IDは保全する。

## [environment](environment/index.md) — 環境の癖

- WSL2のVMが動いたままdistroが再起動する場合がある。旧プロセスの回収はPID 1の開始時刻とプロセス一覧で確認する。
- WSL2のRAM不足で学習や親のbackground workerがtracebackなしで止まった。WSL内の上限と同時に動くプロセスを調べる。
- DINO計測で1秒ごとのnvidia-smi照会がタイムアウトし、監視がworkerを停止した。停止判定の0.17秒前にもDINOのheartbeatが更新されていた。過去のGPU停止の原因は未特定。
- WSL2のPLCS multiview学習では16GB GPUのVRAM上限付近でWindowsホストが短周期に固まった。ピークVRAMに余裕を残す。

## [gotcha](gotcha/index.md) — 落とし穴

- 親CLIのE2Eで識別子の曖昧さが出たら、pytestから実セッションIDを外して同じテストを比べる。
- 実行中のbashスクリプトを同じinodeに上書きすると、途中から変更後の処理を実行する場合がある。一時ファイルから置き換える。
- Claude Codeの会話を再開したら監視プロセスの生存とjobの状態を確かめる。以前の会話が終わると監視も止まった例がある。
- codex-followupのtimerは作成時のスクリプトを絶対パスで参照する。timerが動いている間はそのcheckoutの変更や削除に注意する。
- Codex注釈workerに曖昧な途中終了条件を渡すと短時間で止まった例がある。作業範囲・時間上限・終了条件を具体化する。
- Colabでは修正待ち約1.4時間で割当が消え、VMに取得していた入力を取り直した例がある。
- 汎用のignore規則で新しいlib・runs・JSONが一括stageから抜けることがある。git check-ignoreで確認する。
- NHTのgitlinkと実checkoutのずれは実行版に影響する。submodule内の未追跡設定を整理する前にsceneの参照を確かめる。
- pre-commitのmypyは対象のstagedファイルをfollow-imports=skipで検査する。全体検査と違う型エラーは同じ対象で再現する。
- spin setup-nhtは実行したcheckoutのNHTに共通コマンドを結び付ける。作業用worktreeでは専用の実行パスで試す。
- 共有.venvの依存はgit-syncでは更新されない。統合レビューの起動前に共有環境でjsonschemaをimportできるか確かめる。
- 共有.venvで使われるsrcは実行時のカレントディレクトリで決まる。試すworktreeから実行・enqueueする。
- 共有venvのstandalone Pythonではimport testsがsite-packagesを指す場合がある。pytestと同じimport解決を前提にしない。
