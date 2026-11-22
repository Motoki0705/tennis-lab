# PR #2148 再レビュー（parent-1990-29006c342da84b45a355828a1edcfbc2）の指摘の修正

前回の3件は解消を確認した（修正前HEADで新テスト16件失敗→修正HEADで全件成功、CI全14ジョブ成功、ローカルのconfiguration/storage 6,921件成功）。再レビューで、共通パス解決 `src/utils/configuration/local_storage.py` の `declared_data_path` にmajor 2件が出た。どちらも静的推論なので、**先に再現テストを書いて修正前に失敗することを確かめてから**直すこと。PR本文の閉じ指示は、作業中は `Refs #1990` のまま変更しない（親がレビュー前に戻す）。実機の配置切替・index・hydrate・実削除・unit・sudoは行わない。

## pr-2148-1（line 62）worktree経由の正当なDATA絶対パスが拒否される
指摘: mainをR、worktreeをW、`W/data → R/data`（切替後のworktree手順）とする。DATA rootは `R/data` へ正規化されるが、入力 `W/data/synthetic_data_generation/scenes/B00` はalias側（`R/data/...`）にもE:側にも前方一致せず、Noneを返して包含エラーになる。例: `edit_alignment.py` は `--data-root` だけをresolveし、`--scene-root` はそのまま `NonHydraPathBoundary` へ渡す。
- 求める動作: worktreeの `data` symlink（または同等に、標準DATA rootへ解決される祖先）経由で書かれた絶対パスも、宣言した未管理リンクを通る限り、mainの綴りと同じ検証・同じ論理名で通す。未宣言リンク・誤target・E:未mount・subtreeからの脱出は従来どおり拒否する。祖先の解決を「何でもresolveして一致すれば可」にして検査を弱めないこと（どの祖先の綴りを標準rootの別名として認めるかを明確に限定する）。
- テスト: worktree相当の配置（W/data→R/data、R/dataはD:の実dir、宣言リンクはE:へ）で、consumer経由（少なくとも合成pipelineのscene-root相当と、Clip Studioの1経路）で読めること、および宣言外・誤targetで止まることを検証する。

## pr-2148-2（line 28）E:参照の例外が標準DATA rootに限定されていない
指摘: 標準rootかどうかを basename が `data` であることだけで判定し、任意の `root.parent` をrepositoryとして扱っている。たとえば custom root `[local path]` の中で `synthetic_data_generation` が同じroot内の別dirへのsymlinkなら、従来の包含検査では許可されたのに、今回からE: mountと固定targetの検査で拒否される。逆に、計算上のE:対応先へのリンクならcustom root外への解決を許してしまう。
- 求める動作: 例外は、実際の標準DATA root（このcheckoutの `data`、worktreeではmainの `data` へ解決されるもの、すなわちstorage_tierが管理する配置）に対してだけ適用する。どのrootを標準とみなすかの判定方法を明確に一意にし、READMEの契約に書く。custom DATA rootでは従来の包含規則を完全に維持する（内部リンクは許可し、外部リンクは拒否）。
- テスト: custom rootでの内部リンクの許可と外部リンク（E:対応先の形を含む）の拒否、標準rootでの既存の許可・拒否を両方検証する。

## 完了条件
- 再現テスト（修正前に失敗）、修正、必要ならREADME（3言語）とPR本文の更新。
- 下流への影響を確認する（PathResolverを使うconsumerの既存テスト）。
- 対象HEADのCIで `Python` 成功まで確認する。rate limitが出たらresetを待って再試行。
- 最終報告: 各指摘の再現と修正、テスト件数、CIのURLと最長shard時間。作業時間の上限は2時間。
