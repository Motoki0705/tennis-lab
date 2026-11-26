## parent CLI が確認した待機結果

対象HEADのPR #2462 にmerge conflictがあり、CIが起動していない。同じbranchとPRで最新のmainを取り込んでコンフリクトを解消し、検証してpushしたらPARENT_RESULTで新HEADを宣言する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "ci",
  "pr": 2462,
  "head": "0b12a3fa27d4cdc49f163981a542bcad91499a21",
  "status": "conflict",
  "reasons": [
    "pr_conflict",
    "ci_python_missing"
  ],
  "base_ref": "main",
  "mergeable_state": "dirty"
}
