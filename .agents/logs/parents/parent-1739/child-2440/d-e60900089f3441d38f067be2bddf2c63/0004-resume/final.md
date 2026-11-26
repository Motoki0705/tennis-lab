ドラフトPRを `main` 向けに更新し、#2448 への依存と棄却された疑似正解の採点を外しました。全試合形式の選手選別は `|x| ≤ 5.485 m`、`|y| ≤ 16.885 m`、クリップの25%滞在が既定です。対応付けの人数は `players_per_side` で指定します。[PR #2450](https://github.com/Motoki0705/tennis-lab/pull/2450)

- 原因分析55クリップで、20,804件の段階別集計と選別外れ10,853件を再現しました。
- meiji_4cam全69クリップの保存観測による比較では、3D選手フレームが13,218件増え、停止は14件から11件に減りました。新既定での再選別は、比較時のマスクと全276視点で一致しました。
- Meiji 3camの人手ラベル付き開発用4クリップでは、選別数に変化はなく、誤選別は0件でした。meiji_4camの誤選別率は未測定です。
- pytest 104件、pre-commit、knowledge検査が通過しました。保存済みシーンへの反映は #2460 の担当です。

ブランチ `codex/parent-1739-issue-2440-primary` の HEAD `ceac82ab6efb0db331dbe9cd9490eb271f432bba` をpushし、CIの起動を確認しました。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2450,"head":"ceac82ab6efb0db331dbe9cd9490eb271f432bba","resume_on_success":false}}