# 凍結DINOv3の4条件比較

[定量表・同一画像36パネル](comparison/comparison.md)が比較結果、[provenance](comparison/provenance.json)が照合済み条件と選択checkpointの記録である。元evaluation JSONのSHA-256、選択receipt、各runの記録、Drive保存先、再現コマンドは[run.json](run.json)から辿れる。

## 条件とcheckpoint選択

S/S+/B/Lのbackboneを完全凍結し、LoRAなし、共通後段をTransformer dim1024・8層・16 heads・FFN2752、DPT512に揃えた。共通後段は128,635,436学習parameterで同じ初期重みを使う。S/S+/Bには4段別の学習可能な1×1射影があり、LはIdentityである。射影のparameter数は表に分離した。

B00〜B03とTennisCourtDetector-v1の入力版とsplitを固定し、長辺512・縦横比維持・クロップなし、K変換とpaddingを共通化した。今回の画像は512×288。batch8（合成4・実画像4）、seed42、20epoch、optimizer/loss/precision等を揃えた。各runのlastはepoch19・step33160で、保存されたval/loss最小のcheckpointを選んだ。4条件とも結果としてepoch17・step29844が選ばれた。合成testの値を選択には使っていない。

Lの学習sourceは7d6f53b94f81f8cf7f08b6b78fddcc94d25cc67f、S/S+/Bは75e30a565b1f7a5f7fd4f08cea2daf458d377653である。差分は、正当なbf16 autocast入力を許可するdtype検査修正で、モデル演算と構成は変更していない。Bの評価と比較処理もc4の実装を用いた。比較CLIの2 source fileがc4とbyte一致することを確認した。

## 観測と解釈

合成test886画像ではBのKP平均3.9861pxとSEG mIoU0.95213が最良だった。一方でLがLINE Dice0.65007、semantic LINE mIoU0.46189、pose再投影平均9.58015pxで最良だった。Bはpose回転誤差1.7562度が最小で、Lが並進0.5281mとfocal相対誤差5.47%で最小だった。backboneサイズを大きくすれば全出力が一様に良くなる結果ではない。

実画像val2211画像のKP平均は1.3426〜1.3675pxで、S+とBの差は約0.000044pxだった。SEG/LINE/semantic LINEはLが最良だが差は小さく、この1回だけで一般的な順位や優越性を確定しない。実画像にはpose教師がない。

固定画像は全モデルで同じ合成4件・実画像4件を用い、各headをS/S+/B/Lの順に並べた。代表例の目視では実画像のコート線は4モデルで近い予測を示す。合成の斜視画像にはpose投影のずれが残る例がある。各画像のGTとの対応はパネルで確認し、選んだ8画像を全評価集合の代わりに扱わない。

## 比較の制約

- seed42の1 runずつであり、複数seedや信頼区間は測定していない。既存の採用モデルを自動で置き換えない。
- LはVM消失後に2epoch終了checkpointから再開した。optimizer/schedulerは復元したが、RNG/samplerまで中断なしとbitwise一致する保証はない。
- 実画像valはcheckpoint選択にも使用する。同じsplitの報告値であり、独立testの成績とは扱わない。
- Sのpose平均184.275pxは、同じB01画像の無効depthの2点が誤差総和の91.65%を占めた。元の全点平均を比較表に保持する。深度有効点だけの補足平均15.3926pxで置き換えない。点別証拠はS runのpose-diagnosticにある。
- KP/poseのpxは長辺512へ変換した入力座標。poseパネルはGTを緑・予測を赤で表示し、正のdepthかつ画像内にある投影点だけを描く。描かれない投影点も公式距離指標では所定の可視条件に従って集計する。
- 無効化されているKP-pose consistency距離のraw値0は未測定項目であり、精度として報告しない。終端モデルの自動testも、最良checkpointの正式評価とは分離した。

次に比較を深めるなら、B/Lの複数seedで差の再現性を調べることと、深度が不正になるpose投影の原因・対策を別条件として検証することが有効である。今回の条件や予算を変更した追加学習は行っていない。

## 再現

4 runのevaluation bundleを含むcheckoutのrepo rootで `bash knowledge/runs/group-i983-court-dinov3-shared1024-s42/repro.sh <新しい出力directory>` を実行する。既存の確定済み比較を上書きしない。学習・評価を再実行する場合は各runの設定・入力snapshot・checkpoint選択記録に従い、Colab L4とDrive永続化を用いる。
