論文制作スキル、全11章と付録の設計書、LuaLaTeX＋LuaTeX-ja の雛形を追加しました。`codex/parent-2627-issue-2632-primary` の `f44f9adbe` を push し、ビルド結果を PR に記録しました。[設計書](https://github.com/Motoki0705/tennis-lab/blob/f44f9adbe64fa835f2ed51ee27a888d208df8c86/paper/tennis_scene_architecture/design.md) · [PR #2640](https://github.com/Motoki0705/tennis-lab/pull/2640)

コミット済みファイルから README の手順でビルドし、終了値0、A4・2ページ、最終ログの警告0件を確認しました。一時原稿では図表・相互参照・BibTeX 引用が解決し、未定義の参照と引用を加えると終了値1になりました。既存の文書テスト40件と pre-commit も通過しました。[雛形 PDF](https://github.com/Motoki0705/tennis-lab/blob/f44f9adbe64fa835f2ed51ee27a888d208df8c86/paper/tennis_scene_architecture/build/main.pdf)

停止クリップのうち3件は、同じ2D観測で選別幅を変えると復帰した記録があります。2D検出と停止の因果は #2636、GPT-6.2 Sol による注釈の来歴は #2637 で確認します。絶対3D誤差と横断バッチ単独の反復計時は、追加評価の採否と担当が親 #2627 に残っています。[停止クリップ #2636](https://github.com/Motoki0705/tennis-lab/issues/2636) · [ボール注釈 #2637](https://github.com/Motoki0705/tennis-lab/issues/2637) · [親 #2627](https://github.com/Motoki0705/tennis-lab/issues/2627)

CI はキューに入りました。完了確認を親 CLI に引き渡します。[CI 実行](https://github.com/Motoki0705/tennis-lab/actions/runs/45811541480)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2640,"head":"f44f9adbe64fa835f2ed51ee27a888d208df8c86","resume_on_success":false}}