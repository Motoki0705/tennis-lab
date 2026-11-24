[PR #2498](https://github.com/Motoki0705/tennis-lab/pull/2498) の major 指摘を修正し、HEAD `0d4bf0990f82ae0b8ebf2afe08d3209e583d2364` を push しました。同率 IoU では、検出スコア降順、BBox の辞書順、COCO17 関節配列の辞書順で候補を選びます。スコア閾値で絞った割当にも同じ規約を適用します。

回帰テスト 4 件を追加し、関連 pytest 32 件、pre-commit、knowledge 検査が通過しました。報告された同一 BBox の例では、どちらの行順でも coverage 付き OKS と PCK は `1.0` です。

ViTPose 6,844 画像の再採点は、保存順でもランダムに並べ替えた順でも、公開時の全指標・全集計と厳密に一致しました。公開物 66 件と両 manifest の SHA-256 も公開時の記録と一致しました。PR 本文の実行証拠を更新し、[新 HEAD の CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45401328104) を確認待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2498,"head":"0d4bf0990f82ae0b8ebf2afe08d3209e583d2364","resume_on_success":false}}