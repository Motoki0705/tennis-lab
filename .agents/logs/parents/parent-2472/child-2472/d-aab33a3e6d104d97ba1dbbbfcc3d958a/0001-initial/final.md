Pillar II の日本語 Wiki ドラフト5ページを、指定 worktree の `tmp/wiki_pages/` に作成しました。

| ファイル | 行数 |
|---|---:|
| [Court-Model-Evolution-and-Backbones.md](https://github.com/Motoki0705/tennis-lab/blob/bc671221dcfa490984ca2146ec1ce22916e8887b/tmp/wiki_pages/Court-Model-Evolution-and-Backbones.md) | 96 |
| [Court-Geometry-Postprocessing-and-Video-Shared.md](https://github.com/Motoki0705/tennis-lab/blob/bc671221dcfa490984ca2146ec1ce22916e8887b/tmp/wiki_pages/Court-Geometry-Postprocessing-and-Video-Shared.md) | 97 |
| [Synthetic-3DGS-Alignment-and-Generation.md](https://github.com/Motoki0705/tennis-lab/blob/bc671221dcfa490984ca2146ec1ce22916e8887b/tmp/wiki_pages/Synthetic-3DGS-Alignment-and-Generation.md) | 79 |
| [Synthetic-Scaling-Diversity-and-Pair-Studies.md](https://github.com/Motoki0705/tennis-lab/blob/bc671221dcfa490984ca2146ec1ce22916e8887b/tmp/wiki_pages/Synthetic-Scaling-Diversity-and-Pair-Studies.md) | 74 |
| [NHT-Rendering-and-Foreground-Compositing.md](https://github.com/Motoki0705/tennis-lab/blob/bc671221dcfa490984ca2146ec1ce22916e8887b/tmp/wiki_pages/NHT-Rendering-and-Foreground-Compositing.md) | 58 |

`.venv/bin/python` で構造・本体リンク123件・Wiki 内部リンク13件・秘匿情報・主要な集計値を検査し、エラー0件でした。Mermaid は3図とも構文解析に成功しました。ブランチは `codex/parent-2472-issue-2472-primary`、HEAD は `bc671221d` で、`git status --porcelain` は空です。

- KP512 の **1.708886 px** は validation の再評価値です。配備重み `b863df1f01f0` の学習ノードは未登録で、論文の旧推論に使った重みとはバイト列が異なります。
- 指定写真の H 成立は TCD 公式 **0/4**、argmax 補助 **1/4**、KP＋LINE 共同推定 **4/4**。共同最適化は実装どおり SciPy の `method="trf"` と記載しました。
- Canonical v3 の正式公開 **8,433枚**と論文掲載時の **8,415枚**は、参照したデータ版を明記しました。L4 の checkpoint 公開量 **約43 GB/run** は概算で、Drive のパス解決は遅延要因の候補です。

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}