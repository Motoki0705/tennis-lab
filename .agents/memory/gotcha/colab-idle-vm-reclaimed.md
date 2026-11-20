---
title: "アイドル中のColab VMが回収され、取得済みの入力が消えた"
summary: "ColabのVMがアイドル中に回収され、取得済みの入力を再取得した例がある。接続失敗時は既存割当の有無を確認する。"
applies_to: all
source: "#1585 のColab本実行（2026-10-31）"
created: 2026-10-31
last_verified: 2026-11-20
evidence: "最後のjobの終了から約1.4時間後に `colab.py reconnect` が `Session is absent from Colab registry` を返し、`doctor --remote` の割当が0件だった（VMの開始から約7時間で、約12時間の寿命より前）"
citations:
  - path: ".agents/skills/tennis-colab/scripts/colab_core.py"
    symbol: "Session is absent from Colab registry"
  - path: ".agents/skills/tennis-colab/references/runtime.md"
    symbol: "障害からの再開"
---

2026-10-31には、入力（pool・実画像dataset・重み、計約4.5GB）を取得済みのVMを、コード修正のmergeまで約1.4時間待たせたところ回収され、入力の再取得が必要になった。VMは作成から約7時間だった。`colab.py exec` はSSH切断で失敗し、`reconnect` で割当が消えたと確認した。

**使い方:** 長い修正待ちが見込まれる場合は、入力の再取得を計画に含める。SSH切断で `exec` が失敗したら、`reconnect` と `doctor --remote` でVMの存在を確かめる。異常時は [Colabの復旧手順](../../skills/tennis-colab/references/runtime.md#障害からの再開) に従う。
