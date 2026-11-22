# #1990 第2段: 未分類dataの配置（人間の決定）の反映

手順書の作成ありがとう。親が確認したところ、既定案では切替後に data/ の未分類トップレベルが data/ のパスから消え、コードが直接読む `data/smplx`・`data/smplh`（`src/submodules/vendor/gvhmr/body_model/__init__.py`、`src/tennis_scene/installation_assets.py`、`src/tasks/plcs/visualization/review/service.py`）が壊れる。人間に質問して、次のように決まった（#1990 の合意通知 https://github.com/Motoki0705/tennis-lab/issues/1990#issuecomment-6894034378 が正本。聞き直さない）。

## 人間の決定
1. **コードが読む小さい未分類（ACCAD・samples・smplh・smplx、合計1GB未満）と、tennis_multiview の meiji_4cam は、ハッシュで照合してD:へコピーする。** meiji_4cam は `raw/meiji_4cam` だけでなく `processed/meiji_4cam`（`projects.json` を含む）もコピーする。これらはD:に常駐させ、掃除しない。
2. **それ以外の未分類（tennis_multiview の残り、synthetic_data_generation、tennis_multiview_linux_retained など）は、D:の data/ にE:へのsymlinkを置いて、今のパスを保つ。** storage_tier に、このsymlinkを管理対象外として扱う改修を加える。
3. **outputs/ は、過去のデータをD:へコピーせず、E:だけに残す。** outputs/ はD:の空の実ディレクトリから始める（従来の設計どおり）。
4. tennis_multiview の管理対象化と、WSLの仮想ディスクのsparse化は、別タスクにする（親が改善候補として記録済み。今回は扱わない）。#1990 の容量の受入は、Linux側の空きで判定する。Windows物理D:の差は記録するだけでよい。
5. sudoの手順は、sudo以外の手順をすべて終えてから、親が人間へ通知して依頼する。

## 今回お願いすること
### A. storage_tierの改修（新しいPR）
- origin/main（`54b452531` 以降）から新しいbranchを作る。worktreeは今の `.claude/worktrees/storage-tiering` を使ってよい（merge済みbranchの上で作業しない）。PR本文は日本語、`Refs #1990`（#1990はclose済み。第2段の受入は親が続ける）。
- data/ 直下の未分類トップレベルに置いたE:へのsymlinkを、明示的な設定（config.tomlなど。対象の名前を列挙し、暗黙に全symlinkを許さない）に基づいて、管理対象外のリンクとして扱う。scan・cycle・mirror・evict・hydrate・activate がこれをlayoutエラーにしない。中を辿って単位として発見しない。削除・ミラーの対象にもしない。リンク先がE:の同じrepo相対パスであることを検査し、違えば失敗にする（静かなフォールバック禁止）。E:未mountのときの扱いも明確にする。
- D:へ常駐コピーする未分類（ACCAD・samples・smplh・smplx、meiji_4cam の raw と processed）を、どう扱うか設計する。たとえば、掃除しない常駐の領域として明示的に設定し、ミラー・照合はするが削除しない（ckptと同じ扱い）、または管理対象外の実ディレクトリとして無視する。親の推奨は、ckptと同じ「ミラー・照合のみ、掃除しない」単位にすること（E:との整合を照合できるため）。判断の根拠を報告する。
- テスト（tests README の方針どおり、修正前に失敗すること）、README（3言語）。CIの `Python` 成功まで確認する。このPRは、親が正式レビューにかけてからmergeする。
### B. 手順書の更新
- 上の決定を手順書に反映する（未分類の小物と meiji_4cam の raw・processed の照合コピー、残りの未分類のsymlinkの作成、outputs/ は空から開始）。Clip Studio の利用者への連絡事項（停止・再開の時刻、D:の新しいパスで開けることの確認）も含める。
- 手順書のCLIは、mainのcheckout（現在は23 commit遅れ、他の人の未commitの変更が1件あり、触れない）ではなく、最新のorigin/mainのworktreeから `--repo` でmainのcheckoutを指して実行する形にする。これで成り立つかを確かめる。
- sudoの手順は、sudo以外の全手順の後にまとめる。親がそのまま人間へ渡せる形にする。

## 注意
- 今回も、実機の配置切替・index・hydrate・実削除・unit操作・sudoは行わない。読取専用のdry-runはよい。前回、readlinkでsymlinkのatimeを更新してしまったので、symlinkの一覧は `os.readlink` の代わりに atime を動かさない方法（例: `stat`[local path]`lstat` でのリンク先の取得は不可なので、O_PATH|O_NOFOLLOW のfdから `readlinkat(fd, "")` を使う等）を検討し、無理なら読み取りの回数を最小にして記録する。
- 作業時間の上限は3時間。最終報告: PRのURL、設計の要点、テスト結果、更新した手順書のパス、残る判断事項。
