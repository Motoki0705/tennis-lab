親 #2627 は、テニスシーン3D再構成のアーキテクチャー（標準pipelineの手法＋実行システム）を日本語の技術報告として論文化する。親issue本文・合意と、担当子issueの本文を読む。#2630（閉）の調査コメントと親の統合コメント https://github.com/Motoki0705/tennis-lab/issues/2630#issuecomment-7189580548 に既存記録の数値・出典がある。

人間の判断（2026-11-26）: 各節の主張は人間が示したものを採り、必要な検証で数値の裏付けを作る。検証の結果が主張と異なる場合は、数値を変えずに記録し、最終報告で親へ明示する（親が人間に伝える）。

担当子issue: #2632

## 進め方と報告

- 担当子issueの対象範囲と受入条件に従う。設計書の各節には、人間が示した主張と、#2630の統合コメントにある根拠の出典、未完の検証の子issue番号（親issueの子タスク一覧を参照）を書く。
- `paper/court_robustness/`（report.tex・README・build_paper.py）の組版・evidenceの作り方を読み、再利用できる部分は流用する。`paper/reference/arXiv-2508.10104v1` はメインcheckoutにだけある（git管理外）ので、構成の参照はメインcheckoutのパス（[local path]）から読む。参考論文のファイルはrepoへコピーしない。
- `paper/` は `.git/info/exclude` で無視されるので `git add -f` を使う。
- 雛形のPDFビルドを実行して確かめ、pre-commitを通してPRを作る（本文は日本語、`Closes #<担当子issue>`）。PR作成・push後はCIを `PARENT_RESULT` の wait で宣言して終了する。
