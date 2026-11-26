## parent CLI が確認した待機結果

対象HEADのPR #2469 にmerge conflictがあり、CIが起動していない。同じbranchとPRで最新のmainを取り込んでコンフリクトを解消し、検証してpushしたらPARENT_RESULTで新HEADを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "ci",
  "pr": 2469,
  "head": "5a58fad995290155af9dfa7582f55956b30fbf19",
  "status": "conflict",
  "reasons": [
    "pr_conflict",
    "ci_python_missing"
  ],
  "base_ref": "main",
  "mergeable_state": "dirty"
}
