# Meiji 4cam 外部ボール位置のシーン反映

[選手のみの保存結果](../run-i2363-trt-player-video/README.md#保存先と可視化)に、
外部JSONのピクセル座標から求めたボール3Dを追加しました。
対象は `meiji_4cam/video_000` の `clip_000`〜`clip_068` です。
入力JSON、採否、固定幾何の方針は #1739 で合意しています。 [要件の確定コメント](https://github.com/Motoki0705/tennis-lab/issues/1739#issuecomment-7120448233)

## 入力と実行

- 外部位置は `data/tennis_multiview/processed/meiji_4cam/dataset/videos/video_000/clip_000-068_ball_positions/` の69 JSON。
  `clip_065` の4カメラは入力名に `(1)` が付き、カメラキー・フレーム番号・サイズ・FPSは照合済みです。
- 更新先は `/mnt/wsl/tennis-data/home/kamimura/projects/tennis-lab/.worktree-outputs/parent-2278-issue-2363-primary/i2363/i2363-all-v1/inputs/dataset`。
- 共有向き、共有コートとdriftは #2241 の版を使用しました。向きの参照は配置したJSONの内容digestで固定しました。 [共有向きの再現記録](../run-court-side-video-shared-20261118/reproduce.md#保存済み判定を使う)

実行した `src/` は commit `5f447e4b851ce015feee05ea6121c5e479d39387` に対応します。
代表の `clip_000`（選手あり）と `clip_020`（人物対応停止）を先に公開・検証し、
全件のコマンドでこの2件の公開参照も再検証しました。
CPU 1プロセスで1 clipずつ処理し、コンポーネント配列はmmap、圧縮NPZの検証はメンバーごとのストリーム読み出しを使いました。

同じ実行を計測する場合は、切替前の `i2363-all-v1/inputs/dataset` を指定します。
既に更新したdatasetの元の参照は `restore` で戻せます。 [コマンドと復元方法](../../../src/tennis_scene/pipeline/imports/README.ja.md#外部のクリップ別ボール座標)

```bash
dataset=/mnt/wsl/tennis-data/home/kamimura/projects/tennis-lab/.worktree-outputs/parent-2278-issue-2363-primary/i2363/i2363-all-v1/inputs/dataset
bash knowledge/runs/run-meiji-external-ball-20261124/reproduce.sh "$dataset" outputs/external-ball-reproduction
```

実測の `smoke.json`、`full.json`、`smoke.log`、`full.log` は担当worktreeの
`outputs/external-ball/` にあります。各clipの旧参照と新版、および完全な切替記録は
dataset側の `annotations/tennis_scene/updates/external_ball/` に保存しました。

## 結果

全69 clipで新しいsceneを公開し、`scene.json` と `annotation.json` が同じ新版を読むことを照合しました。
ボール3Dは **45,958/46,228 frame（99.42%）** が有効でした。
欠測はview不足（理由コード1）が270 frame、
幾何solver不成立（理由コード7）が0 frameです。欠測はすべて clip_064 にあり、
同clipの有効数は982/1,252 frameでした。
[clip別の3D有効率・版・秒数](clips.csv)・[集計JSON](summary.json)

55 clipは既存の `camera_alignment` を使い、旧sceneのボール以外の
1,100配列でshape・dtype・NPYのSHA-256が一致しました。
14 clipでは保存済みの校正と共有の半回転 [false, false, true, true] を使いました。
選手軸は0件で、13 clipには人物対応、1 clipにはコート向きの停止理由を記録しています。
この14 clipのカメラ幾何の人物検証は未実施です。

| 入力ステータス | 全点 | 画面外 | 画面内 | 3Dに寄与 |
| --- | ---: | ---: | ---: | ---: |
| `observed` | 121,889 | 3 | 121,886 | 121,840 |
| `interpolated` | 6,476 | 4 | 6,472 | 6,468 |
| `candidate_supported` | 6,835 | 4 | 6,831 | 6,830 |
| `estimated` | 49,712 | 4,125 | 45,587 | 45,587 |

| 採用view数 | 全frame | 有効3D |
| ---: | ---: | ---: |
| 0 | 219 | 0 |
| 1 | 51 | 0 |
| 2 | 774 | 774 |
| 3 | 1,559 | 1,559 |
| 4 | 43,625 | 43,625 |

有効な3Dに寄与した全viewを集計すると、再投影残差の中央値は
4.72 px、95パーセンタイルは 18.00 px でした。

| 計測 | 時間 | ピークRSS |
| --- | ---: | ---: |
| 代表2 clipの初回公開 | 28.20 秒 | 1.19 GiB |
| 全69 clipの公開・再利用確認 | 1132.18 秒 | 2.46 GiB |
| 公開後の独立読戻し | 62.45 秒 | — |

全件のコマンドは代表2 clipの公開を再検証し、残る67 clipを生成しました。
CSVの各clipの `publish_seconds` は初回公開時の値で、合計は
1131.88 秒です。代表2 clipの有効数は clip_000 356/356, clip_020 651/651 でした。

## 再投影の確認

円は外部JSONの位置、十字は保存された3Dを同じカメラへ再投影した位置です。
各図には4 viewが有効かつ画像内に再投影できたframeから、カメラ平均誤差の中央値・95パーセンタイルに
対応する2 frameを示しています。数値は元画像のpixel単位です。

- 既存 `camera_alignment` を使った clip_000。 [233・146 frame](reprojection-clip_000.png)
- コート向きで停止し、既存校正と共有向きを使った clip_012。 [441・46 frame](reprojection-clip_012.png)
- 人物対応で停止し、既存校正と共有向きを使った clip_020。 [149・560 frame](reprojection-clip_020.png)

外部JSONには動画SHA-256とPTSが無いため、外部位置の対応付けにはファイル上のclip・
カメラ識別子、`clip.json` の諸元とフレーム連番を使いました。
保存済み媒体のSHA-256は共有コート側のsourceと照合しています。
再投影誤差は外部位置への当てはまりを示します。独立した3D正解への精度は未計測です。
