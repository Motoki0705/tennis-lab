# PR #2023 HEAD 7819fded4 の親確認で見つかった実機の問題

lease解除時のbreak伝播と、O_PATH・file bindの実測・検出は、親が確認した（修正前のコードで10件失敗、修正後に136件成功、CI成功）。ただし、追加した `Inspector.mount_references` は**この実機では、E:側の検査を全部失敗させる**。今回も配置切替・実削除・unit操作は行わない。

## 親の実測（読取のみ）
- 自分のUIDのプロセスの多く（#1859の学習、bash、codex app-server、dbus など、mnt namespace 4026532228）のmountinfoに、CIランナー用の読取専用bindが2つある。
  ```
  288 83 8:48 [local path] [local path] ro,noatime shared:2 - ext4 [local path] rw
  289 83 8:48 [local path] [local path] ro,noatime shared:2 - ext4 [local path] rw
  ```
  `[local path]` は `drwx--x---+ tennis-actions`、`assets` は `drwx------` なので、`[local path]` へのstatは、すべてPermissionErrorになる（74プロセス中、該当は72件）。8:48 はE:（[local path]）のdevice。
- D:（8:64）のmountで同じdeviceのものは `[local path]`、`[local path]`、`[local path]`、`[local path]` 等。対象外のプロセス（PID 544 systemd --user など）のものもある。
- 現状のコードでは、E:のtreeを検査するとき（verify・mirror更新の `quiet`、旧ミラーの削除）に、device 8:48 が一致する上のmountごとにstatし、PermissionErrorになり、unknownとして失敗する。つまり、実機ではE:側の照合・ミラー更新が常に止まる。

## 求める修正
- mount別名として問題になるのは、**mountのroot（mountinfoの4列目）が単位の内部にあるもの**（単位内のファイル・ディレクトリを別の場所へbindしたもので、inodeへの参照をmount自身が持つもの）だけ。rootが単位の祖先や単位の外にあるbind（上のCI用 `.../data` や `.../ckpt`、`[local path]`）は、通常のパスと同じく単位を親ディレクトリ経由で見せるだけで、単位のentryを消せばそのmountからも見えなくなる。これらはstatせずに対象外にする（件数を監査に記録する）。
  - 単位の「filesystem内のパス」は、検査するプロセスの側ではなく、ツール自身のnamespaceでそのdeviceのmount（E:なら `[local path]` のroot `[local path]`）から求める。quarantine・retained・stageの名前でも判定できること。
  - mountinfoのrootが `[local path]` 付きなど判定できない場合や、単位内のrootなのにstatが失敗した場合は、従来どおりfail closed にする。
  - 判定の境界（rootが単位そのもの／単位内のファイル／祖先／兄弟の単位／同名の接頭辞 `data/a` と `data/ab`）をテストする。上のCI用bindと同じ形（祖先bind、到達不能な親ディレクトリ）でfail closedにならないこともテストする。
- **実機の読取専用dry-runを再実行する。** 小さい単位1〜2個で、D:側の `cycle --dry-run` と、E:側の検査を通る `verify --dry-run` を実行し、unknownが0であること、atime等が不変であることを確かめる（以前の実機dry-runは、この検査を追加する前のもの）。

## 完了条件
- 再現テスト（修正前に失敗すること）、修正、README（3言語）、PR本文。`Closes #1990` は維持する。
- 対象HEADのCIで `Python` 成功まで確認する。
- 最終報告: 修正、テスト件数、実機dry-runの出力の要点。作業時間の上限は1.5時間。
