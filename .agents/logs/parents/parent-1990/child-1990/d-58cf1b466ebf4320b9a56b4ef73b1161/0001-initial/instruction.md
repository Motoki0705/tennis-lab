# #1990 [local path] ストレージ階層の実装（第1段: 実装・テスト・実機dry-run）

#1990 の「全体方針・合意事項」に記録した人間との合意（2026-11-14）を正とする。合意済みの決定を聞き直さない。
今回のターンは **実装・テスト・README・実機の読取専用dry-runとPR作成まで**。
**配置の切替（E:→D:への移行、symlinkの付替え）と実削除は行わない。** それは親が他の親（#1859の学習など）と時間を調整してから、同じsessionへの継続指示で依頼する。

## 合意済みの決定（要約。正本はissue）

- D:（WSLルートext4、relatime）を作業実体、E:（`[local path]`、USB HDD上のext4 vhdx、noatime）を同じrepo相対パスのミラー（正本）にする。
- 管理単位: `data/` はデータセット版ディレクトリ、`outputs/` は run-id ディレクトリ（`outputs/<task>[local path]<purpose>[local path]<experiment>[local path]<run-id>[local path]`、[src/tasks/OUTPUTS.md](src/tasks/OUTPUTS.md)）。`ckpt/` はミラー・照合のみで掃除しない。
- 未使用判定: D:上の単位内全ファイルの最終atimeだけ（2週間）。保護指定の仕組みは作らない。自前ツール（ミラー・照合）の読込みは `O_NOATIME` でatimeを動かさない。
- 照合: ミラー時にD:とE:の両方を全件SHA-256で照合しmanifestに記録。削除直前はD:を再ハッシュしてmanifest一致、E:側はサイズ・mtime・ctimeが記録時から不変であることを確認。
- 取得: 手動CLIでE:→D:へ戻して照合。D:にない単位を参照したら、E:を黙って読まずに取得CLIを促す明確なエラーで止める（どこで検出するかは設計して報告する。全学習コードへの侵襲が大きければ案を出して止まる）。
- 常駐: systemd timerで日次（ミラー→照合→削除）。事前予告の猶予なし。削除記録をjsonl（対象・サイズ・照合結果・時刻）に残し、失敗時だけ通知。dry-runを持つ。
- 消さない条件: E:未mount、照合失敗・不一致、使用中（実行中のtraining queue job が参照、開いているファイル）。E:不在を「ミラーなし」と解釈して進む・照合を省くことは禁止（静かなフォールバック禁止）。
- 新しいworktreeの `outputs/` はD:上に作り、`data/` からのhardlinkが同じfilesystem内で成立するようにする。

## 現状の事実（2026-11-14 親が確認）

- repoの `data/`・`outputs/`・`ckpt/` は `[local path]` へのsymlink（約299GB / 104GB / 29GB）。多くのworktreeの `outputs/` も `[local path]` へのsymlink（一部はメインの `outputs` へ）。
- E: のmountは system unit `tennis-data-mount.service`（`[local path]`）。sudoはパスワードが要るので、新しいunitはrepoの systemd --user unit（`.agents/ops/install_units.sh` の方式、`.agents/ops/cleanup/systemd/` が例）にする。user unitからsystem unitへの順序依存は張れないので、実行時に `mountpoint` で確かめて未mountなら失敗として止まる。
- E:上のatimeは移設時のコピーで11/14にそろっており、noatimeなので以後も更新されない。D:のルートは relatime。D:の空きは約553GB。
- 移行後に新しいworktreeの `outputs/` がE:側にできると、`data/` からのhardlinkが EXDEV で失敗していた。
- #1859 がD:へhardlinkしたデータで学習中。既存のsymlink・データに触れない。

## 配置と既存規約

- コードの置き場所は `.agents/ops/storage_tier/`（cleanupと同じ常駐運用の形）を第一候補とする。別の場所が適切なら理由を報告する。
- 既存の [storage-policy](.agents/skills/tennis-drive/references/storage-policy.md)（Drive）、training queue skill、worktree作成の規約（AGENTS.md・workflow memoryの `.venv` symlink）、`.agents/ops/cleanup` と重複・矛盾させない。[local path]、他からはリンクする（二重管理しない）。
- worktree作成時に `outputs/` をD:に作る規約・仕組みも今回の範囲。

## 設計で明示的に扱って報告すること

- 単位の発見方法（データセット版・run-idの境界をどう決めるか。境界が曖昧なディレクトリの扱い）。
- hardlink（link count > 1。`data/` と worktree `outputs/` 間など）を持つ単位の削除・ミラー・取得の扱い。消しても容量が戻らない・他の参照が残る場合をどう記録するか。
- 書込み中の単位（新しいrunなど）をミラー・照合しないための判定。
- 既存のE:にしかない単位（移行後の初期状態）のmanifestをどう作るか（E:全件読込みの所要時間の見積りを含む）。
- 移行手順（第2段で実行するもの）の具体案: どの単位をD:へ戻すか（`ckpt/` と、使用中・queueが参照する単位）、symlinkの付替え順序、他sessionへの影響と停止が必要な範囲。

## テスト

[tests README](tests/README.md#テスト作成方針)に従い設計してから実装する。少なくとも次を一時ディレクトリ上の擬似[local path]
E:未mount（mountpoint判定の失敗）で何も消さず失敗になる、ハッシュ不一致・E:側のサイズ[local path]、使用中（queue running job参照・open fd）で消さない、2週間未満のatimeで消さない、`ckpt/` を消さない、正常な単位は消えて削除記録が残る、取得で戻した単位が照合を通る、dry-runが何も変更しない。

## 実機dry-run（読取専用）

実機の `[local path]` とD:に対し、変更を伴わないdry-runだけを行う（全件ハッシュはE:に負荷が大きいので、小さな単位1〜2個に限る）。atimeを動かさないことを確かめる。

## 完了条件と報告

- PRを作成し（本文は日本語、`Closes #1990` は付けない。第2段が残るため `Refs #1990`）、CIの対象HEADの `Python` 成功まで確認する。
- 最終報告: PR URL、設計の要点（上の「明示的に扱うこと」への回答）、テスト結果、実機dry-runの出力要約、第2段の移行・実削除の手順案と所要時間・リスク。
- 作業時間の上限は6時間。上の設計事項のうち人間の判断が必要な分岐（合意の範囲外になるもの）が出たら、その時点で選択肢と推奨を書いて止まってよい。それ以外の理由で途中終了しない。
