人物モデル比較 5 ページを本番 Wiki の `master` に公開し、公開後の検証を完了しました。公開コミットは `99b66328f567570ac7cc3077d2bf32c3a4ca7ef2` です。

公開ページはすべて匿名 HTTP 200 でした。[概要・結論](https://github.com/Motoki0705/tennis-lab/wiki/Person-Model-Comparison-Overview) · [評価・計測方法](https://github.com/Motoki0705/tennis-lab/wiki/Person-Model-Comparison-Methods) · [人物検出・ECPose](https://github.com/Motoki0705/tennis-lab/wiki/Person-Detector-and-ECPose-Comparison) · [Top-down 2D pose](https://github.com/Motoki0705/tennis-lab/wiki/Top-Down-Pose-Model-Comparison) · [FastHMR・GVHMR](https://github.com/Motoki0705/tennis-lab/wiki/Body-Recovery-FastHMR-vs-GVHMR)

- **配信**。SVG 34 件・PNG 34 件すべてで HTTP 200、MIME、承認済み画像とのバイト一致を確認しました。削除した一時比較 3 ページは Wiki トップへ HTTP 302 で遷移し、旧原稿 3 件と旧画像 44 件の raw URL は HTTP 404 でした。
- **実表示**。Chromium で 1440×980／390×844 のライト・ダークを全 5 ページで確認しました。既定表示の横スクロールは全 20 条件で **0 件**。全 SVG のデコード、折りたたみの開閉、Mermaid 3 図の描画を確認し、代表の SVG 図表に見切れや文字欠落は見つかりませんでした。
- **狭幅での挙動**。全開時は生成テキスト表 27 件で表内スクロールが発生します。概要の Mermaid では GitHub の操作ボタンが下部ラベルに重なり、図を移動してラベルを読めることを確認しました。

実測とスクリーンショット 59 枚は[公開後検証レポート](https://github.com/Motoki0705/tennis-lab/blob/a4b1250bc7a8ef0e3d0ffd17226b95ad6238a461/tmp/person-revision/published-check.md)にまとめました。

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}