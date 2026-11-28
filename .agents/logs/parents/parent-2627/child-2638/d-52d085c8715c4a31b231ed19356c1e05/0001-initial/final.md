#2638のLaTeX初稿を完成させ、Ready PRを作成しました。[PR #2720](https://github.com/Motoki0705/tennis-lab/pull/2720)

- ブランチ `codex/parent-2627-issue-2638-draft`、HEAD `a874159cb2b61d1cb9e25446a4fd268637969111`。作業ツリーはクリーンです。
- 要旨・全11章・付録・参考文献18件を収録しました。PDFは25ページ、14図・15表です。[生成PDF](https://github.com/Motoki0705/tennis-lab/blob/a874159cb2b61d1cb9e25446a4fd268637969111/paper/tennis_scene_architecture/build/main.pdf)
- `claims.md` は84行で、根拠46行と本文・表の507箇所を照合しました。固定資料97ファイルとコピー図版26ファイルのSHA-256を確認しました。
- コミット済みHEADで `build.sh --check-tracked` は24.09秒、終了値0でした。未解決の参照・引用、文字欠落、Overfull、未追跡入力、最終ログの警告は各0件です。原稿用スクリプトはruffとmypyを通過しました。

保存記録に従い、#2633は同じ4シーンを使った比較としました。#1739の再投影残差は4camの共通55本、予約未見IDF1は3camの結果として記載し、3D時間追跡は計測後に選択式の経路として統合されたことを明記しました。

代表ページの目視確認まで完了しました。PR CIは確認前で、推敲と全ページの最終確認は#2639の工程です。

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}