# PR #2244 正式レビューのmajor指摘への対応（親 #1734 から #2241 へ）

正式レビュー（gpt-6.2-astra / xhigh）はレビューを最後まで終えた。ただし途中で、モデルの容量不足による再接続が起きたため、runnerはruntime_errorとして記録した。指摘の内容は妥当と判断し、次の5件に対応してもらう。修正後に、親が正式レビューをやり直す。親の判断は各項の末尾に書いた。

## 1. 既定の clip モードでも既存 cache と完成 marker が失効する

実行検証。base と HEAD の src 以下の Python ファイルから orchestrator.py:62 と同じ方法で fingerprint を再計算すると、2b712017… から 2d8dd0a9… に変わりました。保存済み clip_000 の court_side で入力・設定・依存参照を固定し、この値だけを替えても execution key は d2b51a33… から f627f16d… に変わります。execution_plan.py はこのキーで通常 cache の再利用と load を判定し、dataset の完成 marker も同じ fingerprint を含みます。共有を選ばなかった利用者にも再計算または明示 overwrite が必要になり、確認点1の cache identity 維持を満たしません。回帰検証には base で保存した store と marker の再利用を含めてください。

親の判断: **直さない（意図した変更として受け入れる）。** src全体のfingerprintはどの変更でも変わる設計で、#2222でも同じ扱いにした。PR本文の「cacheへの影響」に、既定のclipモードでもcacheと完成markerが無効になること、再計算または明示的なoverwriteが必要なことを明記する。

## 2. base で保存した通常の clip 設定を読み直せない

実行検証。base の pipeline.yaml から court_side 節を取り出し、HEAD の _COURT_SIDE_SCHEMA で検証すると、court_side.mode と court_side.video_shared の欠落により MissingConfigurationKeyError になります。configuration.py:253 で両方を必須にしたため、保存済みの合成設定を PipelineRuntimeConfig.from_config に渡す既存の入口は、従来どおりの clip 判定を要求していても起動できません。旧形式の clip 設定を読み込めることと、video_shared を選んだ場合に build/load の指定漏れを拒否することを、両方検証してください。

親の判断: **直す。** 旧形式（court_sideにとの両方がない）の保存済み設定は、clipモードとして読めるようにする。旧形式だと判定できる場合だけを受け入れる。片方だけがない場合と、video_sharedでbuild/loadの指定がない場合は、明示的に拒否する。両方をテストする。

## 3. 同じ orchestrator で build を再実行すると、別 clip の入力変更を見逃す

静的な推論。orchestrator.py:109 のメモリ再利用経路は動画成果物の verify() だけを呼び、build request の内容と各 store の inputs_digest を再照合しません。例えば A・B で一度 build し、B の ball_points を更新してから同じ orchestrator で A の新しい run を開始すると、A 自身の観測照合は通り、更新前の B を含む共有判定が使われます。同じ request を新しい orchestrator に渡せば prepare.py:159 で入力変更として失敗します。追加テストは後者の関数を直接呼んでおり、この差を検出できません。新しい run の build では pin を再照合し、一致した場合に採点結果を再利用してください。

親の判断: **直す。** 同じorchestratorでbuildを再実行するときも、build requestと各storeのinputs_digestを照合し直す。一致した場合だけ採点結果を再利用する。指摘の手順（A・Bでbuild、Bを更新、Aで再実行）をテストにする。

## 4. 一括処理向け API が別の build request で作った判定を受理する

静的な推論。definition.py:129 は action=load のときだけ設定と渡された成果物の digest を照合します。action=build では、同じコート・閾値・サンプリングで、一部の clip だけを証拠にした成果物を、全 clip を指定した request の standard_definition に渡せます。部分集合で build しても全 clip の単独診断は保存されるため、後続の clip 所属検査も通ります。#1752 が使うこの入口では、設定に記録した証拠集合と実際に使った判定が食い違います。準備済み成果物を渡す場合にも、選択 clip と入力版が当該 build request に対応することを検証してください。

親の判断: **直す。** 準備済みの成果物をstandard_definitionに渡すときは、build・loadのどちらでも、選んだclipの集合と入力の版がそのrequestに対応することを検証する。一致しなければ拒否する。#1752の一括CLIが使う入口なので重要。

## 5. court_side の実装を差し替えると共有判定と矛盾時の停止を迂回できる

静的な推論。definition.py:236 で SharedCourtSideModule を選んでも、同ファイルの add() は overrides に court_side があると実装を置換し、node 設定も差し替え先のクラス名だけにします。既存の clip 用 CourtSideModule を渡した場合、単独で確定した矛盾 clip はその単独結果を下流に返せます。一方、run.json には共有判定に基づく failed が記録され、node の cache identity からは共有 digest も脱落します。video_shared とこの差し替えの併用を明示的に拒否するか、共有 digest と矛盾判定を差し替えの外で強制してください。

親の判断: **直す。** video_sharedモードと、court_sideの実装の差し替え（overrides）を一緒に使うことを、明示的に拒否する。

## 共通の条件
- 静かなフォールバックを作らない。既定（clip）の判定の結果は変えない。
- GPUの再実行は不要。meiji_4camの判定（[F,F,T,T]、margin 0.267）が変わる変更をした場合は、保存済みの成果物を使ってCPUで再計算し、確認する。
- CPUの関連テスト、pre-commit、CIを通し、PR本文に各指摘への対応を追記する。merge・closeは親が行う。

