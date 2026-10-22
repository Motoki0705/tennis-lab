# 共有memory SUMMARY

作業を始める前に必ず読む。詳細が必要なときだけ、層の索引から個別のエントリを開く。読み方・書き方・圧縮・削除は [memory skill](../skills/memory/SKILL.md) に従う。memoryは規則ではなく検証待ちの観察で、コードと食い違ったらコードが正しい。

## [workflow](workflow/index.md) — 作業のコツ

- 新しいworktreeには `.venv` が無いので、メインtreeの `.venv` へsymlinkを張る。

## [decision](decision/index.md) — ユーザーの判断

- 判断を仰ぐときは、既に答えが出ている事項を聞き直さない。選択肢には、コードを読まずに判断できる説明を付ける。品質確認は動画で見せる。
- 長時間ジョブは定期的に起きて確かめず、完了・失敗だけを監視して待つ。
- 古いckptは、指標がknowledgeにあれば削除してよい。
- 注釈workerは gpt-6.1-sol / max。テスト用の値を本番planの実行条件に流用しない。
- 公開向けの文書は英語版を既定にし、日本語版を `.ja.md` に分ける。文書にハッシュ値を書かない。

## [environment](environment/index.md) — 環境の癖

- WSL2 + 16GB GPU。VRAMがあふれてもOOMにならず、ホスト全体が固まる。ホストRAM（上限24GB）が尽きると、学習はtracebackなしで止まる。 Blender EEVEE は d3d12 の環境変数を指定しないと、黙って CPU で描画する。

## [gotcha](gotcha/index.md) — 落とし穴

- `src` はカレントディレクトリから import される。worktreeのコードは、worktreeで実行・enqueueしないと使われない。
- pre-commit の mypy は staged ファイルだけを検査する。`.gitignore` の汎用パターンが新しいディレクトリを黙って無視する。`tests/` はパッケージではない。
- 実行中のbashスクリプトを上書きしない。codex-followup の timer は作成時のスクリプトを絶対パスで実行する。
- submodule の pointer がPRで進んでも、メインtreeの checkout は古いまま残る。`spin setup-nht` は実行したtreeに全体の `nht-render` を結び付けるので、メインtreeでだけ実行する。

## [review](review/index.md) — マージ可否の判断基準

- 独立した目的の変更は別PRに分ける。
- 名前は機能・方式が伝わるものにし、Issue番号は追跡情報に置く。
- 特定の実験だけを進めるためのコードは常設のsrcに残さない。再現に必要なものはknowledgeに残す。
