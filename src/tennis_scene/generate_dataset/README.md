# Sceneの増分生成

5入口のコマンド、入出力ツリー、rootの指定、再実行方法は
[tennis_sceneの入出力契約](../README.md#2-generate_dataset--datasetの各clipへsceneを追加する)を正本とします。

`manifest.py` はdataset/clip manifestの唯一の定義です。
`pseudo_annotation.py` は各clipの動画をrunnerへ渡し、frame数・FPS・解像度・必要配列を検証します。
一時ディレクトリでarchiveと設定を保存し、完成マーカーを最後に作って公開します。
既存結果の置換はbackupを伴うトランザクションで行い、途中状態を成功として扱いません。

ここでは学習用split、RGB特徴、教師の補正・採否判定を生成しません。
SLCSの学習準備は [SLCS生成処理](../../tasks/slcs/generate_dataset/README.md) が担当します。
