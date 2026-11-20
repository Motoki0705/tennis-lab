---
title: "Colabで複数条件を学習するときは、1セッションで逐次に実行し、VMの消失から再開できるようにする"
summary: "Colabの複数条件の学習は1セッションで順に実行し、VMを作り直しても保存済みの段階とcheckpointから再開できるようにする。"
applies_to: all
source: "2026-10-30 のユーザー指示（Court Detectionのシーン組合せ比較の計画）"
created: 2026-10-30
last_verified: 2026-11-20
evidence: "ユーザー指示を後続の実行条件と照合し、VM消失時の再開方法は tennis-colab の runtime.md で確認する。"
citations:
  - path: ".agents/skills/tennis-colab/references/runtime.md"
    symbol: "障害からの再開"
    note: "VM消失後に使える復旧経路"
---

ユーザーは、複数条件の学習をColabで行うとき、L4の1セッションで1本ずつ順に実行することを望む。条件ごとに並列のVMを立てるとdatasetの取得を繰り返すため。VMの寿命やアイドル回収に備え、保存済みのcheckpointと段階ごとの完了記録から再開できるようにする。

**使い方:** Colabの複数条件の実行を設計するときは、1セッションでの逐次実行と、新しいVMで同じコマンドを再実行すれば完了済みの段階を飛ばし、途中の学習を同じ条件のcheckpointから再開する手順を最初から含める。並列化を提案するなら、データ取得の重複というコストを明示する。
