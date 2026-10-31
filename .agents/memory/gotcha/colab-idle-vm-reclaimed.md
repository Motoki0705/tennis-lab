---
title: "ColabのVMは、jobを実行せずに待たせると数時間の寿命より前に回収され、取得済みの入力も消える"
applies_to: all
source: "#1585 のColab本実行（2026-10-31）"
created: 2026-10-31
last_verified: 2026-10-31
evidence: "最後のjobの終了から約1.4時間後に `colab.py reconnect` が `Session is absent from Colab registry` を返し、`doctor --remote` の割当が0件だった（VMの開始から約7時間で、約12時間の寿命より前）"
---

ColabのVMは、managed jobを実行していない状態が続くと、約12時間の寿命を待たずに回収される。2026-10-31には、入力（pool・実画像dataset・重み、計約4.5GB）を取得済みのVMを、コード修正のmergeまで約1.4時間待たせたところ回収され、入力の再取得が必要になった。`colab.py exec` は `Connection closed by UNKNOWN port 65535` で失敗し、`reconnect` で不在が分かる。

**使い方:** 修正待ちなどでVMを1時間以上使わない見込みなら、取得済みの入力を残すためにVMを待たせず、全jobの保存を確認して `stop` する。入力の再取得を前提に計画する。`exec` がSSHの切断で失敗したら、`reconnect` と `doctor --remote` でVMの存在を確かめてから新しいsessionを始める。
