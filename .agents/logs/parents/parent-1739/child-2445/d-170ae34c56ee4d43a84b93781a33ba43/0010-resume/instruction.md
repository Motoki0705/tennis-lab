## parent CLI が確認した待機結果

対象HEADのPR #2469 にmerge conflictがあり、CIが起動していない。同じbranchとPRで最新のmainを取り込んでコンフリクトを解消し、検証してpushしたらPARENT_RESULTで新HEADを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "ci",
  "pr": 2469,
  "head": "f8add8f0444ce8581cf3d266f6eee91716b4c5ee",
  "status": "conflict",
  "reasons": [
    "pr_conflict",
    "ci_python_missing"
  ],
  "base_ref": "main",
  "mergeable_state": "dirty"
}
