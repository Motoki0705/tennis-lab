[PR #2023](https://github.com/Motoki0705/tennis-lab/pull/2023)を更新しました。branch `codex/storage-tiering`、commit `7819fded488ba03068ea25c7b2ed4642f9d391dc`。作業ツリーはcleanです。

- **majorは5ケースで再現・修正済み。** `F_UNLCK`直前の状態と、解除時に回収したSIGIOを失敗へ伝播します。`phase=after_unlink, unlinked=true`をjournal・auditへ残し、transactionを保持します。
- **O_PATH・file bindをWSL2 6.6で実測しました。** 両方ともnlink=1でlease取得可能で、unlink・解除後にnlink=0のinodeへ再openして書けました。既存O_PATHは従来のFD照合で検出済みです。別namespaceのfile bindは見逃しを再現し、mountinfoと`[local path]`経由のdevice/inode照合で検出するよう修正しました。

実測コマンドは次です。2件成功し、参照取得前はbusy=false、取得後はtrue、解除後のreadbackは`pear!\n`でした。

```bash
.venv/bin/python -m pytest -n 0 -s \
  tests/integration/agents_ops/test_storage_tier.py \
  tests/e2e/agents_ops/test_storage_tier_mounts.py \
  -k 'opath_reference or external_file_bind_reference' -q
```

**14ケース増、関連609件成功（45.26秒）**。故障注入・復元後検証、pre-commitも成功しました。[対象HEADのPython成功](https://github.com/Motoki0705/tennis-lab/actions/runs/43642394772/job/132597886533)、CI全14ジョブ成功を確認しています。integration 123件はCIで全件成功。実mount 8件はローカル成功、CIでは権限制約でskipです。

残る前提は、使用中検査後にO_PATH/proc-FD/bind参照を新規取得しないことです。競合区間を数マイクロ秒以内とは保証しません。検査可能な所属プロセスのないnamespaceも一覧化できません。README 3言語へ明記しました。

既存データの配置切替・実削除・unit操作は未実施です。`Closes #1990`は維持し、統合確認・mergeを親へ引き継ぎます。