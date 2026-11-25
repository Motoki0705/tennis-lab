## parent CLI が確認した待機結果

対象HEADのPython CIが成功した。宣言した後続作業を行う。
固定疑似BBoxでViTPose-Hのflipあり・なしの精度採点を共有GPU queueへ登録する。TRT16はflip設定とprofileが一致するengineを照合する。速度には #2496 の基準値を使い、補助担当の成果を同じPRへ統合する。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "ci",
  "pr": 2500,
  "head": "33ae11e6ff421a61d85aac139fa7a0fbfab1da39",
  "status": "success",
  "reasons": []
}
