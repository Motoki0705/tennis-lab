## parent CLI が確認した待機結果

対象HEADのPR #2610 にmerge conflictがあり、CIが起動していない。同じbranchとPRで最新のmainを取り込んでコンフリクトを解消し、検証してpushしたらPARENT_RESULTで新HEADを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "ci",
  "pr": 2610,
  "head": "cba442a2d9aeb21d413133aa421770e627a8755f",
  "status": "conflict",
  "reasons": [
    "pr_conflict",
    "ci_python_missing"
  ],
  "base_ref": "main",
  "mergeable_state": "dirty"
}
