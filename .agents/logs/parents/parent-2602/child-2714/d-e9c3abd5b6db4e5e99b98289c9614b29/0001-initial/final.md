#2714 の公開前検証を完了しました。`tmp/wiki/` の人物モデル比較 5 ページから冒頭の定型欄を撤去し、出典・参照日・判断日を本文と出典節に移しました。Typst 表 25 組と Matplotlib グラフ 9 組を配置しています。

| ページ | SVG・PNG | 容量 | Markdown 表 旧版 → 草稿の既定表示 |
|---|---:|---:|---:|
| 概要・結論 | 8 | 0.55 MiB | 5 → 0 |
| 評価・計測方法 | 4 | 0.32 MiB | 8 → 0 |
| 人物検出・ECPose | 16 | 2.53 MiB | 9 → 0 |
| Top-down 2D pose | 26 | 2.91 MiB | 9 → 0 |
| FastHMR・GVHMR | 14 | 1.46 MiB | 7 → 0 |
| **合計** | **68** | **7.77 MiB** | **38 → 0** |

- `verify --rerender --wiki-page` は全 5 ページ・34 組で通過しました。固定正本、転記ブロック全文、画像 SHA-256、画像形式、公開禁止情報の検査を含みます。[検証ログ](https://github.com/Motoki0705/tennis-lab/blob/a4b1250bc7a8ef0e3d0ffd17226b95ad6238a461/tmp/person-revision/verify-wiki.log)
- 旧版 831 行、参照 244 件、見出しアンカー 52 件を照合し、情報欠落・リンク切れは 0 件でした。[移行監査](https://github.com/Motoki0705/tennis-lab/blob/a4b1250bc7a8ef0e3d0ffd17226b95ad6238a461/tmp/person-revision/migration-audit.md)
- Chromium で本文幅 896・390・326px のライト／ダークを比較しました。既定表示で横スクロールが必要な箇所は、順に **2 → 0、22 → 0、25 → 0**。全 SVG、折りたたみの開閉、Mermaid 3 図を確認し、代表画像の目視で文字欠落や見切れは見つかりませんでした。展開時は、生成テキスト表のうち 390px で 17 件、326px で 27 件に表内スクロールが残ります。[表示比較・スクリーンショット](https://github.com/Motoki0705/tennis-lab/blob/a4b1250bc7a8ef0e3d0ffd17226b95ad6238a461/tmp/person-revision/render-check.md)

本番 Wiki への公開と公開後の実表示確認は、Tier 1 の人間確認ゲート承認後の工程です。

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}