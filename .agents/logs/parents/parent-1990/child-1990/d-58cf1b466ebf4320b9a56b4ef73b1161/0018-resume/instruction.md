# PR #2148（旧 #2143）のレビュー指摘の修正

親からの連絡: #2143 は、親CLIの成果記録が委任branchとの一致を要求するため、同じHEAD `f889cde7c` を委任branch `codex/storage-tiering` へfast-forwardでpushし、**PR #2148** として出し直した（#2143 はclose済み）。worktree `.claude/worktrees/storage-tiering` は親が `codex/storage-tiering`（origin追跡）に切り替えてある。以後はこのbranchとPR #2148で作業する。PR本文の閉じ指示は、作業中は `Refs #1990` のままにして変更しない（親CLIの継続の検査のため、親が一時的に変えた。レビュー前に親が `Closes` へ戻す）。

正式レビュー（parent-1990-05a898c66ef944ebb5c99cca894e5aae）でmajor 3件が出た。静的確認なので、**先に再現テストを書いて失敗を確かめてから**直すこと。今回も実機の配置切替・index・hydrate・実削除・unit・sudoは行わない。

## pr-2148-1（config.toml:16）許可したE:リンクが、PathResolverのresolve後の包含検査で拒否される
指摘: D:の data を実ディレクトリにすると、許可したE:へのリンクは `PathResolver.resolve_beneath` のresolve後の包含検査（`src/utils/configuration/paths.py:253–263`）で、root外として拒否される。たとえば、Clip Studioの `tennis_multiview/raw/meiji_3cam/video_000` や、合成pipelineの `synthetic_data_generation/raw/B00.mp4` が読めなくなる。従来はdata rootの全体がE:へ解決されていたので成立していた。
- 求める動作: PathResolverが、storage_tierの設定で宣言した `unmanaged_data_links` に限って、リンク先がE:の同じrepo相対パス（ミラー上の対応パス）である場合だけ、そのリンクを辿った解決を許す。宣言の正本はstorage_tierのconfigの1か所にし、PathResolverへ二重に書かない。宣言外のリンク、リンク先の不一致、E:未mountは、従来どおり明確なエラーにする。`src/utils` の変更なので、下流への影響を確かめ、テストは必須。
- テスト: 実際のconsumer（Clip Studioの入力解決、合成pipelineの入力解決など、代表の2つ以上）を通して、切替後の配置（D:の実ディレクトリ＋宣言したE:リンク）で読めることと、宣言外・不一致で止まることを検証する。

## pr-2148-2（references.py:34）列挙外のdata symlinkを拒否していない
指摘: `data_parents` は宣言したパスの祖先だけを返すので、dataset_containers自体を検査しない。たとえば `data/ball_detection/v1` が未許可のsymlinkでも、`inspect_references` は見逃し、`discover` の `directories` はunmanagedに分類するだけになる。scan・activate・引数なしのcycleが成功し、本来管理する版が黙ってミラーの対象から外れる。
- 求める動作: container直下も、リンクを辿らずに検査し、未許可・誤ったリンク先を拒否する。data/ 配下のどこであっても、宣言外のsymlinkが単位の境界や上位にあれば失敗にする。

## pr-2148-3（engine.py:193）dry-runの前に許可リンクをreadlinkし、atimeを更新する
指摘: dry_runの分岐より前に `inspect_references` を呼び、`references.py:88` で許可リンクをreadlinkする。古いリンクではatimeが更新されるので、scanや無関係の単位への `--dry-run` でもD:のmetadataを変える。
- 親の判断: symlinkのリンク先を読むと、Linuxは（readlink・O_PATHからのreadlinkat・パス解決での追従のいずれでも）symlink自体のatimeを更新し得るので、完全には避けられない。宣言した未管理リンクのatimeは、未使用の判定（単位内のファイルのatime）にも、他の判定にも使わない。よって、**dry-runの無変更の契約から「宣言した未管理リンクのsymlink自体のatime」だけを明示的に除外**し、READMEの契約に書く。そのうえで:
  - 単位のpayload・管理対象のmetadata・状態ファイル・ディレクトリは、dry-runで一切変えないことをテストで検証する（変更の記録だけでなく、無変更を検証する）。
  - リンクを読むのは、必要な操作（宣言リンクの検証が要る操作）に限り、無関係な単位への `--dry-run` では読まない、など読み取りを最小にする。最小にできない場合は理由を書く。
  - この除外が妥当でない根拠（たとえば、そのatimeを何かが使っている）を見つけたら、報告して止まる。

## 完了条件
- 再現テスト（修正前に失敗すること）、修正、README（3言語）、PR本文の更新。
- 対象HEADのCIで `Python` 成功まで確認する。GitHub APIのrate limitが出たら、resetを待って再試行する。
- 実機の読取専用QAを再実行する（unknown 0）。
- 最終報告: 各指摘の再現と修正、consumer経由の検証の内容、テスト件数。作業時間の上限は2.5時間。
