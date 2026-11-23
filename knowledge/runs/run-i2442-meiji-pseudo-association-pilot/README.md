# Meiji の人物対応疑似正解 / Terra 試行

2026-11-24 に `codex exec`、`gpt-6.1-terra`、`xhigh` で試行した。Meiji 3cam の人手参照で較正し、meiji_4cam の停止・1視点欠測・確信層で実行と使用量を確認した。詳細な入力・プロンプト・応答・ラベルの SHA-256、使用量と集計は [pilot-results.json](pilot-results.json) にある。 [担当 #2442](https://github.com/Motoki0705/tennis-lab/issues/2442)

## 固定入力

[#2363 の all-v1](../run-i2363-trt-player-video/README.md) から採用済みの検出、追跡、選別、姿勢、人物対応、三角測量、カメラ校正・向きと進捗診断を選び、69 clip の同期 RGB と manifest を複製した。全276本の同期 RGB は元の scene index の SHA-256 と一致した。4本の収録元動画は [raw-videos.json](raw-videos.json) で識別する。

- worktree の複製は `outputs/meiji-pseudo-association/inputs`。作業用実体の削除後も残る複製は、E:の repo ミラー直下の `outputs/player_association/evaluate/meiji_4cam/i2442-pseudo-association/inputs` にある。
- `SHA256SUMS` は12,184件、manifest 自体の SHA-256 は `b380d77c8ee80b9a9d0faaaab1c7cceccd3f8f8eb446e1b23b3d51224cdda62a`。作業先とは別ディレクトリで `sha256sum --quiet --check SHA256SUMS` が終了値0。
- 検出・追跡・選別とローカル校正は69 clip、確定した court side は68 clip、確定した人物対応・三角測量は55 clip。停止層は選手不足の020・058と、選手境界が曖昧な002・004・014・018・023・028・036・038・039。11件とも court side と暫定対応診断があり、確定対応の手前で停止した。別理由の012（court side 停止）、049・060（区間間の割当が曖昧）は保存済み段階の観測で抽出した。

## 69 clip の抽出

[抽出設定](../../../src/tasks/player_association/configs/pseudo_annotation_meiji.yaml) に基づく。異なる保存先で全件を再実行し、`selection.json` はバイト一致した。SHA-256 は `be9108d27454148e35537597606c2937c6786d9327a4d87006ae89c29c837b7e`。保存先を変えた1回は29.30秒。

| 層 | 検出のある候補 frame | 調査用の抽出 frame | 人の抜き取り案 |
| --- | ---: | ---: | ---: |
| 選別外れ・コート近傍 | 39,152 | 399 | 30 |
| 1視点・3D欠測 | 4,247 | 71 | 28 |
| 未割当・再投影不一致 | 2,130 | 187 | 30 |
| ID switch・断片 | 12,281 | 350 | 30 |
| 停止 clip | 9,006 | 66 | 22 |
| 確信層から無作為 | 3,709 | 144 | 30 |
| 重複を除いた合計 | 46,101 | 1,205 | 170 |

同期するsource frame 全体は46,228。コート近傍の選別外れは同じ人の長時間の観測や選手以外の候補も拾うため、候補は広く分布した。層別の上限を適用すると69 clip から10〜26 frameずつ、全source frame の2.61%になる。確信層は旧経路の支持が高い frame の対照で、全体からの無条件無作為標本とは分母が異なる。

## 精度の較正と4cam 試行

Meiji 3cam の人手確定4 clip から各12 frameを選んだ。番号とbox 座標だけを人手ラベルから取り、役割・人物ID・説明は入力から外した。人が判定できたboxの内訳は下表の通り。人物IDは clip 全体・カメラ共通の一度の最適な置換で合わせた。box一致は役割と選手IDが一致したもの。人物ID単位は同じカメラ・frame・人の重複boxをまとめ、すべて一致したもの。

| 3cam clip | box 一致 / 人が判定できたbox | 人物ID単位の一致 |
| --- | ---: | ---: |
| video_000/clip_000 | 72 / 72 | 72 / 72 |
| video_000/clip_007 | 72 / 72 | 72 / 72 |
| video_001/clip_001 | 92 / 94 | 63 / 65 |
| video_002/clip_013 | 84 / 84 | 72 / 72 |
| 合計 | 320 / 322（99.38%） | 279 / 281（99.29%） |

人の参照が曖昧な5 box は一致率の分母から除いた。相違2件は ID switch を含むシングルスの区間。較正は歴史的な開発用4 clip の既存観測boxに対する測定で、meiji_4cam の層別誤り率は人のレビューで測る。

| 4cam clip | 示した情報 | frame | box / 曖昧 | CLI 経過秒 |
| --- | --- | ---: | ---: | ---: |
| 020 | RGB、検出番号 | 6 | 173 / 6 | 59.512 |
| 032 | RGB、検出番号 | 6 | 187 / 3 | 52.694 |
| 025 | RGB、検出番号 | 4 | 126 / 0 | 35.451 |
| 032 | RGB、検出番号、raw track と旧選手ID | 同じ6 | 187 / 0 | 45.334 |

clip_025 は旧経路で生成され、1視点欠測層が0 frame、未割当・再投影層が1 frame、確信層が125 frame。4cam の各試行で4選手に割当があった。clip_032 の同一box の比較では、旧経路を伏せた判定が明瞭だった184 box のうち178件で一致し、選手単位は93/96一致。旧IDを示す精度上の利点はこの比較では判定できないため、旧経路との比較に使う参照には独立した画像判定を採った。旧IDを伏せた3 clip・16 frame・486 box のラベルを固定 dataset の `annotations/player_association/labels.json` に保存した。旧IDを示した出力も比較用receiptとともに保管した。

## 使用量と人の確認案

| 条件 | CLI 秒 / frame | 入力 token / frame | 出力 token / frame | USD 相当 / frame |
| --- | ---: | ---: | ---: | ---: |
| 3cam の4試行 | 2.14 | 3,299 | 84 | 0.00372 |
| 4cam の旧IDなし3試行 | 9.23 | 5,411 | 435 | 0.00481 |
| 4cam の旧IDなし、キャッシュなし換算 |  |  |  | 0.00759 |

秒数はCLI起動から終了までの和をframe数で割った。3cam の48 frameは102.917秒、4cam の旧IDなし16 frameは147.657秒。受領したCLIイベントには token 内訳があり、残りquotaや実請求額の値は得られなかった。金額は OpenAI の Standard token レートでの換算（入力・キャッシュ入力・出力は1M tokenあたり USD 1.00・0.05・5.00）。 [ChatGPT Work / Codex Rate Card](https://help.openai.com/en/articles/20001415-chatgpt-rate-card-enterprise-token-based-pricing)

69 clip、1,205 frame を同じ単価と所要で外挿すると入力約652万・出力約52.4万 token、USD 5.79相当（試行のキャッシュ比率）〜9.14相当（キャッシュなし）。CLIの累計は約3.09時間、4枠を常に使えた場合は約46分。準備、再試行、枠の端数、rate limit待ちと人の確認時間は別途かかる。試行のframe数と、全体案の1依頼あたり10〜26 frameには差がある。

[170 frame の抜き取り案](proposed-human-sample.json) は先に別clipから1 frameずつ選び、必要な層では同じclipの3 frame以上離れた点を2件目とした。停止層は11 clip、1視点層は抽出済み区間の分布で件数が決まった。review では同期4視点でboxの役割とclip内IDを照合し、層ごとに誤ったbox・確認できたbox・人でも判定できないboxを集計する。全体注釈の対象と予算、確認件数はユーザー判断待ち。

## 再現と確認

以下の `E_REPO` は E: の repo ミラー実体を指す。現物の保存先と作業用snapshotを照合し、必要なら [freeze_i2363.py](freeze_i2363.py) で #2363 の元の `i2363-all-v1`、raw `meiji_4cam` ルート、作業用snapshotの順に渡して再作成する。

```bash
study=knowledge/runs/run-i2442-meiji-pseudo-association-pilot
root="$E_REPO/outputs/player_association/evaluate/meiji_4cam/i2442-pseudo-association"
singles=data/tennis_multiview/processed/meiji_3cam/dataset
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export OPENCV_FFMPEG_CAPTURE_OPTIONS='threads;2'
(cd "$root/inputs" && sha256sum --quiet --check SHA256SUMS)
.venv/bin/python -m src.tasks.player_association.pseudo_annotation extract \
  "$root/inputs" outputs/meiji-pseudo-association/selection-repeat.json
cmp "$root/selection.json" outputs/meiji-pseudo-association/selection-repeat.json
.venv/bin/python -B "$study/pilot.py" "$root/pilot" 3cam "$singles"
.venv/bin/python -B "$study/pilot.py" "$root/pilot" 4cam "$root/inputs" "$root/selection.json"
.venv/bin/python -B "$study/summarize.py" "$root/pilot" "$singles" "$root/selection.json" \
  outputs/meiji-pseudo-association
.venv/bin/python -m src.tasks.player_association.scripts.review_dataset \
  --dataset-root "$root/inputs/dataset" --artifact-root "$root"
```

`pilot.py` は受理済みreceiptがある場合、入力・応答・ラベルを再検査する。4cam の公開分はreviewのHTTP経由で3 clipの注釈済みframe、4カメラのRGB、未注釈frameの表示を検査した。詳細なCLI境界は [モジュールのREADME](../../../src/tasks/player_association/pseudo_annotation/README.md) にある。
