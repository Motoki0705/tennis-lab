---
title: "Colabの `exec` 直後に接続が切れると、jobは `pending` のまま起動せず、`status` は `transport_alive: false` を返す"
applies_to: all
source: "#1585 のColab本実行（2026-10-31 の job `campaign-r1`）"
created: 2026-11-01
last_verified: 2026-11-01
evidence: "提出receiptは `status: submitted` / remote `pending` だったが、ローカルのjobの `transport.log` は約13秒後に Colab CLI の `RuntimeError: Connection was lost.` で終わり、Driveの `operations/colab/<session>/` にそのjobの保存が無かった。VMは何も実行せずに回収された"
---

`colab.py exec` の提出receiptが返っても、直後にColab CLIのkernel接続が切れると `remote.py run` は実行されない。remoteの状態は `pending` のまま変わらず、エラーにも終了にもならない。`pending` を「実行中」とみなす監視は、この未起動を検知できない。2026-10-31には、これを約25時間見落とし、その間にアイドルのVMが回収された。

**使い方:** 提出後は `status` で `transport_alive: true` と `running` への遷移を確かめる。監視の終了条件には、jobの終了に加えて、`transport_alive: false` と、数分以上 `pending` から進まない状態を含める。未起動なら、同じVMで新しいjob idに再提出する。
