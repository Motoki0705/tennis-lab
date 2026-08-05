# Incoming raw candidates

このディレクトリは3本のcollector queue branchでのみ使用します。mainまたはdaily PRへraw JSONを直接追加しません。

```text
knowledge/literature/incoming/<JST-date>/<collector>/<timestamp>-<slug>.json
```

各ファイルは `schema/candidate.schema.json` に従うappend-only入力です。GitHub Actionsが検証後、canonical recordを `knowledge/literature/candidates/` へ生成します。
