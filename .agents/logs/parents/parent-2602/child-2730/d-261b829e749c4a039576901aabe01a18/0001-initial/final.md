Pillar III・VI の担当11ページを改訂し、公開前検証を完了しました。表は **43→23件**、Tier 2 の7ページで本文幅390pxの横スクロールは **16→0件** になりました。

| 改訂ファイル | 表数（旧→新） |
|---|---:|
| [tmp/wiki/Ball-Detection-Overview.md](https://github.com/Motoki0705/tennis-lab/blob/a4b1250bc7a8ef0e3d0ffd17226b95ad6238a461/tmp/wiki/Ball-Detection-Overview.md) | 4→0 |
| [tmp/wiki/Ball-2D-Detector-Baseline-and-Domain-Adaptation.md](https://github.com/Motoki0705/tennis-lab/blob/a4b1250bc7a8ef0e3d0ffd17226b95ad6238a461/tmp/wiki/Ball-2D-Detector-Baseline-and-Domain-Adaptation.md) | 6→2 |
| [tmp/wiki/Ball-Refiner-2D-Failure-and-3D-Transition.md](https://github.com/Motoki0705/tennis-lab/blob/a4b1250bc7a8ef0e3d0ffd17226b95ad6238a461/tmp/wiki/Ball-Refiner-2D-Failure-and-3D-Transition.md) | 5→3 |
| [tmp/wiki/Ball-3D-Smoothing-and-Scene-Evaluation.md](https://github.com/Motoki0705/tennis-lab/blob/a4b1250bc7a8ef0e3d0ffd17226b95ad6238a461/tmp/wiki/Ball-3D-Smoothing-and-Scene-Evaluation.md) | 3→2 |
| [tmp/wiki/Ball-Long-Sequence-and-Architecture-Exploration.md](https://github.com/Motoki0705/tennis-lab/blob/a4b1250bc7a8ef0e3d0ffd17226b95ad6238a461/tmp/wiki/Ball-Long-Sequence-and-Architecture-Exploration.md) | 3→4 |
| [tmp/wiki/Research-Models-PLCS-BLCS-SLCS.md](https://github.com/Motoki0705/tennis-lab/blob/a4b1250bc7a8ef0e3d0ffd17226b95ad6238a461/tmp/wiki/Research-Models-PLCS-BLCS-SLCS.md) | 6→3 |
| [tmp/wiki/Glossary-and-Search-Traps.md](https://github.com/Motoki0705/tennis-lab/blob/a4b1250bc7a8ef0e3d0ffd17226b95ad6238a461/tmp/wiki/Glossary-and-Search-Traps.md) | 3→0 |
| [tmp/wiki/Ball-Datasets-Annotation-and-Test-Contracts.md](https://github.com/Motoki0705/tennis-lab/blob/a4b1250bc7a8ef0e3d0ffd17226b95ad6238a461/tmp/wiki/Ball-Datasets-Annotation-and-Test-Contracts.md) | 2→1 |
| [tmp/wiki/Research-and-Infra-Overview.md](https://github.com/Motoki0705/tennis-lab/blob/a4b1250bc7a8ef0e3d0ffd17226b95ad6238a461/tmp/wiki/Research-and-Infra-Overview.md) | 2→1 |
| [tmp/wiki/Installation-Storage-and-Outputs.md](https://github.com/Motoki0705/tennis-lab/blob/a4b1250bc7a8ef0e3d0ffd17226b95ad6238a461/tmp/wiki/Installation-Storage-and-Outputs.md) | 4→3 |
| [tmp/wiki/Knowledge-Library-and-Annotation.md](https://github.com/Motoki0705/tennis-lab/blob/a4b1250bc7a8ef0e3d0ffd17226b95ad6238a461/tmp/wiki/Knowledge-Library-and-Annotation.md) | 5→4 |

- `tools.wiki_assets verify` は11ページすべて通過しました。冒頭定型欄と公開禁止情報は0件です。公開検査が検出した既存のローカルパスは、公開済みの規約から確認できるようにしました。 [CLI ログ](https://github.com/Motoki0705/tennis-lab/blob/a4b1250bc7a8ef0e3d0ffd17226b95ad6238a461/tmp/rollout-ball-research/verify.log)
- 旧版のリンク329件、見出しアンカー63件、数値・母数・比較条件を保全しました。相対リンク48件はすべて解決しました。 [移行監査](https://github.com/Motoki0705/tennis-lab/blob/a4b1250bc7a8ef0e3d0ffd17226b95ad6238a461/tmp/rollout-ball-research/audit.md) · [監査 JSON](https://github.com/Motoki0705/tennis-lab/blob/a4b1250bc7a8ef0e3d0ffd17226b95ad6238a461/tmp/rollout-ball-research/audit.json)
- 代表3ページを Chromium の896px・390px、ライト・ダークの4条件で目視しました。折りたたみの開閉と Mermaid の描画を確認し、スクリーンショット36枚を保存しました。代表3ページの横スクロールは896pxで1→0件、390pxで6→0件です。長系列と用語集は390pxで本文が縦に伸びています。 [実測値とスクリーンショット](https://github.com/Motoki0705/tennis-lab/blob/a4b1250bc7a8ef0e3d0ffd17226b95ad6238a461/tmp/rollout-ball-research/render-check.md)

作業ブランチは `codex/parent-2602-issue-2730-pillar-ball-research` です。公開 Wiki の `master` が進んだ後も、担当11ページの旧版が比較基点と一致することを確認しました。本番への反映と公開 Wiki 全体の最終確認は統合担当に引き渡します。

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}