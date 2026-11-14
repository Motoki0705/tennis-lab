# 役割分担の変更（親から。前回の指示のうち「終了条件（安定確認）」を置き換える）

ユーザーの指示で、学習中の長い見届けは子ではなく親が行うことになった。途中で止めたのはこのためで、あなたの作業に問題があったわけではない。中断前の状態（qualification a3/a4の失敗の調査、Eへのコピー、checkpointingの実装など）から、そのまま続けてほしい。

## 子が行うこと（変更なし）

実装・テスト、短いGPU probe/qualification（自分で起動し、結果まで確認してよい）、Eへのコピーの照合、PRの作成とCI `Python` 成功の確認（待つときは `gh run watch` など、出力を何度も読まない方法を使う）、本番の起動。

## 新しい終了条件

次を満たしたら、学習の完了を待たずに終了する。
1. 本番の直列コマンドを、push済みcommitから、セッションと切り離したsystemd user service（共有queue resource=all）として起動した。
2. T1が5,500から再開し、compileの後に **最低10回の更新** が、finite loss/grad/LR、連続したupdate番号・sample位置で記録されたことを実ログで確認した。peak VRAM（allocated/reserved）とhost availableの最小値が、allocator上限・RAM下限の内側にあることも確認した。
3. 親が監視するための引継ぎ情報を、`knowledge/runs/<run-id>[local path]` とコミットした小さなJSON（例: `monitoring-handoff.json`）に残した。内容:
   - systemd unit名、queue job ID、本番のPID/PGID、終了時に必ず書かれるterminal event（完了・失敗・process終了）のファイルパス
   - 進捗の正本ファイル（serial status、stageごとの最新state/log、Driveへの保存receipt）のパス
   - 親が節目で確認すべきこと（T1 6,000の終端と12k固定validationの保存・readback、T2の600→601のCNN解除、T2最初の500更新checkpointのreadback、各条件の完了）と、それぞれをどのファイルのどの値で判定するか
   - 異常時の停止状態の見方と、再開の正確なコマンド（新しいattempt番号の決め方を含む）

T1の12k評価やT2の節目まで待たない。終了後に定期的な進捗確認やfollow-upを登録しない（監視は親が行う）。

止めてよい条件は前回の指示のとおり。最終報告には前回の項目に加え、上の引継ぎ情報の場所を書く。
