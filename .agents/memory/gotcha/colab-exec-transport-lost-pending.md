---
title: "Colabの `exec` 直後に接続が切れると、jobは `pending` のまま起動せず、`status` は `transport_alive: false` を返す"
applies_to: all
source: "#1585 のColab本実行（2026-10-31 の job `campaign-r1`、2026-11-02 の job `campaign`）"
created: 2026-11-01
last_verified: 2026-11-02
evidence: "提出receiptは `status: submitted` / remote `pending` だったが、ローカルのjobの `transport.log` は約13秒後に Colab CLI の `RuntimeError: Connection was lost.` で終わり、Driveの `operations/colab/<session>/` にそのjobの保存が無かった。VMは何も実行せずに回収された"
---

`colab.py exec` の提出receiptが返っても、直後にColab CLIのkernel接続が切れると `remote.py run` は実行されない。remoteの状態は `pending` のまま変わらず、エラーにも終了にもならない。`pending` を「実行中」とみなす監視は、この未起動を検知できない。2026-10-31には、これを約25時間見落とし、その間にアイドルのVMが回収された。

逆に、remoteが `running` になった後の `transport_alive: false` は異常ではない。この値はローカルの提出clientの生存だけを示す（[tennis-colab の runtime.md](../../skills/tennis-colab/references/runtime.md)）。2026-11-02には、起動を委任したCodex子の終了時に共通ランナーが残った提出clientを回収し（`turn_end_reason: descendants_reaped`、終了値1）、以後 `running` のまま `transport_alive: false` になったが、学習は続いていた。

**使い方:** 提出後は `status` で `transport_alive: true` と `running` への遷移を確かめる。`pending`・`starting` のまま `transport_alive: false` になったか、数分以上 `pending` から進まなければ未起動とみなし、同じVMで新しいjob idに再提出する。`running`・`finalizing` での `transport_alive: false` は監視の終了条件にせず、remoteのstatusで完了・異常終了を見る。起動を子に委任したときの終了値1（`descendants_reaped`）だけで失敗と判断せず、親が `status` を取り直して確かめる。
