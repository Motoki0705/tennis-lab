# 再開（親 #2492 から）

## 宣言したlocal jobの結果

2件とも exit 1 で失敗した。

- `20261124T113216Z_8364d66c2532_rtdetrv4-l-prepare`（112 s）
- `20261124T113216Z_ef583d5731c4_rtdetrv4-x-prepare`（159 s）

ログは `[local path]` にある。どちらも末尾は次のとおり。

```
RuntimeError: mmap can only be used with files saved with `torch.save(_use_new_zipfile_serialization=True), please torch.save your checkpoint with this option in order to use mmap.
```

ログ全体を読み、原因（公式重みの保存形式と読込側の `mmap=True` などの不一致）を特定して直すこと。公式重みを別の形式に保存し直して使う場合は、元の重みのsha256・変換手順・変換後のsha256を記録し、元の重みと数値が一致すること（state_dictの全tensorの一致）を確認する。別の重みに差し替えない。

## 土台の更新

#2495 のPR #2498 と #2496 のPR #2497 が更新された。親がこのbranchへ両方の最新headを取り込んだ（merge commit 4f8d5c657。取り込んだheadは b1011e406 と d64962a28）。#2496 には固定速度入力の扱いが追加されている可能性がある。#2495 には代表6 clipの生成結果が `knowledge/runs/run-meiji-multiview-player-evaluation/` に入った。速度計測と精度採点はこれらを使う。

土台はまだdraftで、今後も更新される。次に待機（`PARENT_RESULT` の waiting）で終えるときは、事前に `git fetch origin codex/parent-2492-issue-2495-primary codex/parent-2492-issue-2496-primary` で両方の最新headを取り込み、テストを通してから終えること。待機明けの自動再開は、両PRの最新headが含まれていないと拒否される。
