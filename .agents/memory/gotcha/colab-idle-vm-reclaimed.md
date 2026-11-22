---
title: アイドル中のColab VMが回収され、取得済みの入力が消えた
summary: Colabでは修正待ち約1.4時間で割当が消え、VMに取得していた入力を取り直した例がある。
applies_to: all
source: '#1585 のColab本実行（2026-10-31）'
created: 2026-10-31
last_verified: 2026-11-20
evidence: 2026-10-31、最後のjobから約1.4時間後にreconnectでregistryから所有sessionが消失したと判定され、doctor --remoteの割当数も0件だった。
citations:
- path: .agents/skills/tennis-colab/scripts/colab_core.py
  symbol: Session is absent from Colab registry
- path: .agents/skills/tennis-colab/references/runtime.md
  symbol: '#障害からの再開'
---

2026-10-31、pool・実画像dataset・重み（計約4.5GB）を取得済みのColab VMをコード修正のmergeまで約1.4時間待機させた。作成から約7時間の時点で `exec` がSSH切断で失敗し、`reconnect` で割当の消失を確認した。新しいVMでは入力の再取得が必要だった。

修正待ちが長引く場合は、入力を再取得する時間も見込む。所有割当の確認と復旧は [Colab runtime](../../skills/tennis-colab/references/runtime.md#障害からの再開) を参照する。
