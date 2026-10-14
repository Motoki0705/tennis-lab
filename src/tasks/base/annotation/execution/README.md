# 資源管理付き工程実行

`StageRunner` は、取得・復号・描画・推論などの親所有コマンドを、固定planと
`ResourceRequest` に結び付けて実行する。具体的な操作は呼び出し元が渡す。
baseからball/playerのモデル・ラベル実装をimportしない。

- CPU/I/O工程は、受付済みleaseと工程lockのFDを専用supervisorへ継承する。
- GPU工程はmain repositoryの共有training queueへ、launch hashをidempotency keyに
  `resource=all` で登録する。queue scriptのpath/hash、provider/session/issueを固定する。
  queue内で初めて資源leaseを取得し、queueが作ったprocess groupを維持する。
  GPU工程自身による `setsid` や別queueへの分離は認めない。
- planのcommit/diffを起動時に照合し、未追跡のソースがあれば拒否する。入力artifact、
  コマンド、timeout、資源要求は `campaigns/<campaign>/jobs/<request-id>/launch.json`
  に保存する。実行途中の設定変更は別のrun plan/明示的な試行にする。
- thread数とGPU可視性を設定し、TMPDIR・Torch・拡張module・CUDA・XDG cacheを
  固定rootの `cache/runtime/` 以下へ置く。副生成物も保存予算に含める。
- timeout/cancelでは、PID再利用を検査した所有子孫へTERM→KILLを送り終了確認する。
  queueのshell祖先は停止対象から除く。stdout/stderrと終了receiptを不変保存し、
  lease解放までは `draining` とする。
- receiptのないsupervisor終了・queue失敗を成功にしない。完了結果は再実行せず回収し、
  失敗後の再実行には新しいrequest IDを要求する。

`src.tasks.base.annotation.execution` CLIはsupervisor用で、継承FDまたはqueue所有権を
必要とする。ユーザー向けの入口は[composition root](../../../annotation/README.md)。
合成コマンドによるテストはtimeout・子孫終了・受付枠・queue group・改変検知を確認する。
GPU reservationを使うテストもsleep/printだけであり、CUDA実測とは区別する。
