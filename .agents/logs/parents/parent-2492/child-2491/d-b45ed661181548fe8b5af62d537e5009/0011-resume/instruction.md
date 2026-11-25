# #2491 統合担当: PR #2501 の正式レビューの major 3件を直す

PR #2501（HEAD `06b4dbf64`）の正式レビューは major だった。レビュワーは、標準経路の宣言・保存先・キャッシュ互換 manifest、#2536 の単眼 PLCS、#2499 の依存接続、固定 commit、資産8点、#2534 の速度・TRT16 の記録、動画24本の SHA-256 を照合し、問題はなかった。親は次の3件をすべて採用する。

なお、PR 本文の closing 対象を親が変えた。子 #2558（FastHMR 経路のコードと計測・可視化を、#2491 の採否と分けて main に入れる）を作り、本文の先頭を `Closes #2558` と `Refs #2491, …` にした。正式レビューは、閉じる子 issue がないと受け付けないため。この本文の形は保つ。

## 1. FastHMR の明示 load が、別精度でインポートした成果を受け付ける
`definition.py` 519行目付近の FastHMR ノードは、保存済み成果の出自を制限していない。共通の `bind_import` と `publish_import` で PT32 の `FastHMROutput` を `origin=import` として登録すると、`execution.fasthmr=load` の PT16 設定でも `execution_plan.py` 276行目付近の検査を通る。実行 identity（モデル設定を含む）の比較は `origin=component` のときだけで、publication の系譜検査も backend を照合しない。
- FastHMR の load で component が生成した成果を要求するか、インポート時に backend を含む完全な identity を検証する。どちらにするかは、既存の他ノードの import の扱いと揃うほうを選び、理由を PR 本文に書く。
- 別精度のインポートを拒否する回帰テストを足す。

## 2. 推論対象が空のとき、固定重みの SHA-256 の不一致を見逃す
計画と publication は資産の実測 SHA-256 を identity に入れるが、固定値との照合はモデルの初期化時にしか行わない。選手の復元リクエストが空だと、`fasthmr.py` 170行目付近はモデルを読まないので、SHA-256 が違う YOLO や diffusion の重みがあっても v3 成果を保存でき、部分シーンも公開できる。
- 固定 SHA-256 との照合を、計画・publication で資産を束縛するときに行う。
- 復元リクエストが空でも、不一致を明示的に拒否する回帰テストを足す。

## 3. VAE の初期化が、設定と違う GPU に重みを復元する
固定版 FastHMR のコンストラクタ（`upstream.py` 103〜106行目付近から呼ぶ）は、VAE を `map_location` なしで読み込む。重みの保存先が `cuda:0` なので、`device=cuda:1` でも GPU 0 の空きメモリに依存する。また、設定で受け付ける `pt32` の CPU 実行が、CUDA が見えない環境では初期化に失敗する（レビュワーが実ファイルで再現した。`map_location='cpu'` なら297テンソルを復元できた）。
- VAE を CPU に復元してから、指定のデバイスへ移すロード契約を作る。上流のコードは固定 checkout なので、直接書き換えずに、こちら側の境界（ラッパー）で扱う。
- CUDA が見えない状態での CPU 初期化と、指定デバイスへの配置を守る回帰テストを足す（GPU がなくても動くテストにする）。

## 終わり方
1. 各指摘の回帰テストを足し、修正前に失敗し修正後に通ることを確かめる。関連テスト、既定 `standard` の回帰テスト、pre-commit を通す。
2. 同じ branch に push する。PR 本文の検証の節を新しい HEAD に合わせて直す。本文の先頭（`Closes #2558`）は変えない。
3. 最終回答には、指摘ごとに直したこと（commit とテスト）を書く。新しい HEAD の CI を待つ `PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2501,"head":"<新HEAD>","resume_on_success":false}}` を返す。親が修正を照合して、マージ処理を続ける。
