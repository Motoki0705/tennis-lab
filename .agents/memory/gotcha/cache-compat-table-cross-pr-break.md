---
title: 互換表の対象ファイルを触るPRは、PR CIが通ってもマージ後のmain CIを落とす
summary: 互換表は対象ファイルの after ハッシュを固定する。別PRがその対象ファイル（例 inventory.py）を変えると、PR CIは古いmainで通り、マージ後にmainの互換テストが落ちる。
applies_to: all
source: '2026-11-27 #1722 の PR #2613 のマージで main CI が失敗し、#2672（PR #2673）で回復した（親セッションの観察）'
created: &id001 2026-11-27
last_verified: *id001
evidence: main で各互換表の files の after と、対象ファイルの dual_sha256 を比べる。不一致の表は適用されず、test_fp32_cache_compatibility.py の main 識別テストが落ちる。
citations:
- path: src/tennis_scene/pipeline/source_identity.py
  symbol: def _compatible_patch
- path: tests/integration/tennis_scene/test_fp32_cache_compatibility.py
  symbol: def test_multiview_integration_preserves_main_source_identity
---

- 互換表は、対象ファイルの全ハッシュが `after` と一致するときだけ適用される。1つでも違うと静かに適用されず、識別は基準版に戻らない。
- main の識別まで固定するテスト（`test_multiview_integration_preserves_main_source_identity` など）がある表では、後から対象ファイルを変えると main CI が落ちる。`src/utils/configuration/inventory.py` は CLI 境界の登録で多くのPRが触るので、特に起きやすい。
- PR CI は作成時点の main とのマージで走るので、互換表を入れたPRが後からマージされると、この組み合わせを検出できない。マージ前に、変更ファイルが main の `src/tennis_scene/pipeline/*_cache_compatibility.json` の対象に含まれるかを確かめる。
- 直すときは、変更が互換であることを表の持ち主に確かめたうえで、その項目の `after` だけを現在の値に更新する（後続マージで `after` を更新してきた運用に合わせる）。main の識別を固定するテストが無い表は、適用されないままでも CI は通る。この状態は、その変更でキャッシュが無効になる安全側の挙動である。
