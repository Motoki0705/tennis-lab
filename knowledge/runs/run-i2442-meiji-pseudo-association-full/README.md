# Meiji 4cam の人物対応疑似正解の実行と棄却

## 棄却判断

2026-11-24 午後、ユーザーが生成結果を見て、meiji_4cam の疑似正解ラベルを品質不足として棄却した。ラベルと棄却前の暫定採点値は評価の根拠から外した。選手選別の #2440 では疑似正解による採点を外し、過去の追跡方式を再評価する #2443 は取り消した。 [親issueのユーザー判断](https://github.com/Motoki0705/tennis-lab/issues/1739) [#2442の判断記録](https://github.com/Motoki0705/tennis-lab/issues/2442#issuecomment-7120884740)

170 frameの抜き取り対象を選び、review APIから全件を読み出せることを確認した。人による抜き取り確認は未実施のまま終了し、4camの層別誤り率は未測定。 [抜き取り計画](../run-i2442-meiji-pseudo-association-pilot/proposed-human-sample.json)

## 方式と実行規模

#2363 の固定成果物から疑義5層と確信層の無作為標本を seed `2442` で抽出した。対象frameごとに同じ時刻の4視点の全景と番号付き検出boxのcropを1枚に合成し、1 clip分を一度に `codex exec` の `gpt-6.1-terra` / `xhigh` に渡した。追跡・選別・対応の旧IDは伏せ、検出boxにclip内の選手ID、選手でない人物、曖昧のいずれかを付けた。 [抽出と試行の記録](../run-i2442-meiji-pseudo-association-pilot/README.md)

69 clip・1,205 frame・34,664 box（曖昧328 box）のラベルをローカルdatasetに保存した。69件とも初回の形式検査を通過し、形式不合格・未処理は0 frameだった。実行ログ上の同時処理は最大4 clip、同一clipの重複実行は0件。集計JSONの `accepted` は形式受理、`published` はdatasetへの書き出しを示す。 [clip別集計](results.json)

| 層 | 保存 frame | 人の確認計画 frame |
| --- | ---: | ---: |
| 選別外れ・コート近傍 | 399 | 30 |
| 1視点・3D欠測 | 71 | 28 |
| 未割当・再投影不一致 | 187 | 30 |
| ID switch・断片化 | 350 | 30 |
| 停止 clip | 66 | 22 |
| 確信層から無作為 | 144 | 30 |
| 合計 | 1,205 | 170 |

2026-11-24 03:42:24〜04:10:45 JSTに実行した。バッチの経過は28分21秒（1,701.183秒）、CLI稼働の合計は5,953.299秒（4.94秒/frame）。CLIイベントで入力3,713,178 token（キャッシュ分1,055,488）と出力300,823 token（推論分119,084）を記録した。試行時のStandard単価でUSD 4.21相当（USD 0.00350/frame）、全入力をキャッシュなしとした場合はUSD 5.22相当。実請求額は未取得。 [使用量と実行時間](results.json)

## 較正と限界

Meiji 3cam のシングルス4 clip・48 frameでは、人手ラベルとの一致がbox単位で320/322（99.38%）、人物ID単位で279/281（99.29%）だった。人が判定できた既存boxを採点した。4camは選手が2人から4人に増え、全人物の検出boxを入力した。3camの一致率がこの条件でも得られるかは未確認。 [較正の内訳](../run-i2442-meiji-pseudo-association-pilot/README.md#精度の較正と4cam-試行)

保存RGBの全景を各視点704×396に縮小し、元画像から切り出したcropを88×126の枠に収めた。遠くの選手は元のRGBで占める画素が少なく、cropを拡大しても元画像の細部が上限となる。選手が重なる場面や、とびとびの対象frameをまたいで外見の似た選手を追う場面も誤りの要因になり得る。各要因の寄与は未測定。検出されなかった人物は注釈の対象外だった。 [当時の画像構成](https://github.com/Motoki0705/tennis-lab/blob/7c1ad6d989f0ed6c6bd9ef81e6347753fa43349f/src/tasks/player_association/pseudo_annotation/images.py)

再試行する場合は、4camの少数clipで人手参照を先に作り、疑義層と確信層の両方で役割・人物IDの一致を測る。遠方のboxを高解像度のcropで別に提示する条件と、対象frameの前後を連続して提示する条件を、同じ人手参照で比較すれば、全体に広げる前に改善の有無を確かめられる。

## 保存先と元の実行手順

生成物は E: の repo ミラーのローカルパス `outputs/player_association/evaluate/meiji_4cam/i2442-pseudo-association/` に保存した。

| 対象 | 保存先からの相対パス |
| --- | --- |
| 検出・追跡出力と同期RGBの固定入力 | `inputs/` |
| 抽出frameと層 | `selection.json` |
| 試行のパケット、応答、receipt | `pilot/` |
| 全体実行のラベル | `full/dataset/videos/<video>/clips/<clip>/annotations/player_association/labels.json` |
| 全体実行のパケット、応答、receipt、実行ログ | `full/clips/<video>/<clip>/`、`full/sessions/`、`driver-20261124.log` |
| 集計・抜き取り計画・機械的な読み出し検査 | `full/results.json`、`full/human-review.json`、`full/verification.json` |
| 記録専用化前のGit履歴 | `repro/pr-2448-original.bundle` |

固定入力の `SHA256SUMS` の SHA-256 は `b380d77c8ee80b9a9d0faaaab1c7cceccd3f8f8eb446e1b23b3d51224cdda62a`、`selection.json` は `be9108d27454148e35537597606c2937c6786d9327a4d87006ae89c29c837b7e`。ラベルには注釈frame、層、モデル・effort、プロンプトと入力の指紋を保存した。ローカルの `full/results.json` の SHA-256 は `3a27b03bda2a6e30879eec622db1570b8fc364b0c31adf2c22f4df485fedc33c`。

注釈コードとreviewの変更は未マージ。全体実行時のCLIとラベル実装は `7c1ad6d989f0ed6c6bd9ef81e6347753fa43349f`。起動時のスクリプトは `full/driver-source/e2272ad199b19b541b6cbe3410aeb516e4f140a24aae7dc418eded9ccad47933/annotate.py` に保管した。起動と再集計の手順は、記録専用化前のPR #2448のHEAD `c1c9b06ab3ebf36b595066bc34377e85c561fd8b` にある。 [元の全体実行README](https://github.com/Motoki0705/tennis-lab/blob/c1c9b06ab3ebf36b595066bc34377e85c561fd8b/knowledge/runs/run-i2442-meiji-pseudo-association-full/README.md)

Git bundleのSHA-256は `d44de3581520b89422328ba3c5d7c03f95426acf9abdb0759164c7b76dcb74c4`。土台の `8d63f4382fb2ff38dd701bbdd99677b971a1d57f` を含むcheckoutから、`git fetch "$root/repro/pr-2448-original.bundle" HEAD` で元のHEADを取得できる。ここで `root` は上記の生成物ディレクトリの絶対パス。
