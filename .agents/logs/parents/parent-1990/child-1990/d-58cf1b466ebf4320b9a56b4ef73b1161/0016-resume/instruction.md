# #1990 第2段の実行計画（調査と手順書。今回は実行しない）

PR #2023 は8回目のレビューでapproveされ、mainへmergeした（merge commit `54b452531`）。ここから第2段（実機の移行・実削除・取得・容量確認・timer導入）に進む。**今回は読取専用の調査と手順書の作成だけで、配置切替・index・hydrate・実削除・unit操作・sudoの操作は行わない。** 親が手順書を確認し、人間とタイミングを合わせてから、同じsessionへ実行を指示する。

## 制約（他の親 #1859 との合意。親が調整済み）
- #1859 の学習が終わる（2026-11-15 21〜24時 JST の見込み）まで、worktree `ball-arch-desktop` の `outputs`、E:のmount、home直下の `tennis-hot`、`.training_queue/` には触れない。
- repo rootの `data/`・`outputs/`・`ckpt/` のsymlink切替は、学習中でもよい（学習はrepoの `data/`・`ckpt/` を読まず、出力もE:上の実体パスで直接開いている）。ただし、E:への大量の読込み（index・hydrate）は、学習のE:への書込み（checkpoint）と競合し得る。学習中に行うか、終了後に行うかを手順書で判断材料とともに提案する。
- CIランナーの再インストールなどsudoの手順は人間が実行する。runnerと `tennis-lab-training-queue.service` の停止は、queueが空のときだけ（学習を止めない）。

## 調べること（読取専用）
1. mainの最新（`54b452531` 以降）で、storage_tierのCLIを使って実機のdry-runを行う。`scan`、未分類サイズ、`cycle --dry-run` の出力の要点（配置未切替のエラーなど）。
2. 移行時にD:へ戻す単位を決めるための材料。現在、repoの `data/`・`outputs/`・`ckpt/` 配下の単位を開いている自分のUIDのプロセス（fd・cwd・mmap）と、他のworktree・sessionがシンボリックリンク経由で参照している単位。各単位のサイズ。`ckpt/` 全体のサイズ。D:の空き容量（現在値）。
3. 既存worktreeの `outputs` のリンク先の一覧（E:の独立outputs、mainのoutputsへのalias、`ball-arch-desktop`）と、それぞれの移行方法。
4. E:のindex（初回のmanifest作成）の所要時間を、小さな単位で実測して見積る（読取専用。全件は行わない）。hydrateの所要時間の見積り。

## 手順書に含めること
- 実行の順序（各ステップのコマンド、所要時間、検証方法、失敗時の戻し方）。少なくとも次を含める:
  - E:のindex（どの単位から、どの順で。全件を一度にやるか、必要な単位から段階的にやるか）
  - repo rootの `data/`・`outputs/`・`ckpt/` のsymlinkの付替えと、D:の実ディレクトリの作成
  - `ckpt/` と、使用中の単位のhydrate
  - worktreeの `outputs` の付替え（`ball-arch-desktop` は #1859 の学習が終わってから）
  - 使い捨ての単位での、実削除→D:の空きの増加の確認→E:から取得して使えることの確認（受入条件(6)）
  - systemd user timerの導入・有効化（`install_units.sh` の方式）と、最初の日次実行の確認
  - CIランナーのsudoの手順（人間が実行。いつ、どのコマンドを、学習・CIへの影響）
- 他のsession・parentへの影響（切替の瞬間に、D:に戻していない単位を参照するとhydrateを促すエラーになる）と、事前に知らせるべき相手・内容。
- 人間に判断してもらう必要がある点（あれば。合意済みの事項は聞き直さない）。

## 完了条件
- 最終報告に、調査結果の要点と手順書（そのまま実行できる粒度）を書く。作業時間の上限は1.5時間。
