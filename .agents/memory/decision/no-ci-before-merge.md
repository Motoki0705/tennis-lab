---
title: "merge前にCIを起動せず、CIをmergeの条件にしない"
applies_to: all
source: "2026-10-24の本会話（子#1496への指示）と #1349 の合意記録 https://github.com/Motoki0705/tennis-lab/issues/1349#issuecomment-6384460840"
created: 2026-10-23
last_verified: 2026-10-24
evidence: "#1349 方針7・同合意記録と本会話を照合する。導入状態は親の報告、daily_ci status、実timer/runで別途確認し、合意だけを稼働証拠にしない"
---

ユーザーは、push/PRや各mergeを全体CIの起点にせず、毎日06:00以上09:00未満（JST）の保守枠へ集約することに合意した。初回全体CIは毎日1回、修正merge後の新しいmain SHAへの再CIは同じ枠内で許可する。失敗日の未検証範囲を日付変更で捨てない。

検証・修復中は通常mergeを止め、当該cycleに登録した修正PRだけ既存の正式review/gateで進める。全対象の回復を確認したら早期解除し、遅くとも09:00には未復旧でも通常mergeを再開する。未復旧・未検証・停止不明を成功とせず記録・報告し、次回へ持ち越す。日中のローカル検証と正式review、HEAD・差分・依存などの通常gate条件は維持する。専用memory PRも通常経路へ統合し、PR CIを待つ例外を残さない。

**使い方:** 子の全コミット件名に `[skip ci]` を付け、pytest・ruff・mypy等のローカル検証をPRへ記録する。各merge後のCI確認を個別に要求せず、[日次cycle](../../ops/daily_ci/README.md) の対象版と証拠で確認する。本会話では、親 #1349 に全実装・検証・正式review・merge・最後のtimer導入まで明示承認された。子 #1496 は実装・検証・PRまで、実CI起動・merge・timer有効化は親が接続検証後にまとめて行う。この記録は運用開始・稼働済みを示さない。
