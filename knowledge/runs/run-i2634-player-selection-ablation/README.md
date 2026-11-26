# 保存済み観測による選手選別の比較

[#2634の事前設計](https://github.com/Motoki0705/tennis-lab/issues/2634#issuecomment-7190663054)に従い、検出・追跡・特徴を共有して、選別の規則だけを変える。

| 条件 | 候補 |
| --- | --- |
| `all_tracks` | 全raw IDの実観測、上限なし |
| `image_2d` | 実観測率25%以上のraw IDを、観測率 × 中心重み × box高さ重みで順位付け |
| `court_raw_dwell` | 同じraw IDの足元をz=0へ逆投影し、コート領域内の実観測率25%以上を滞在数で順位付け |
| `court_standard` | コート座標とCLIPで断片を連結し、領域内の異なる実観測フレーム数をもとにチェーンを選ぶ現行規則 |

画像2Dの中心重みは、box中心の横座標の中央値を画像幅で割り、中央で1、端で0.5とする。高さ重みはbox高さの中央値を画像高の15%で割り、0〜1に収めて0.5〜1へ写像する。画像2Dとコート滞在のraw ID条件は、同点ならraw IDの小さい順に選び、上位6件を元のトラック軸の順に渡す。コート領域はシングルス幅 `|x| ≤ 4.115 m`、ベースライン余白5 mを含む `|y| ≤ 16.885 m`。現行は25%以上の領域内実観測を持つチェーンを最大6件選ぶ。選ばれたraw IDまたは断片の実観測を使い、領域外を走ったフレームも保持する。GSI補間はマスクで除外する。現行のグループでは引き継ぎで重なった観測を下流用の1 boxに絞り、元のraw行を保持する。

## 入力と指標

3camの開発4本は#964 run 11の重複統合offとr7の全画面DINO .30、ViTPose、CLIP特徴。元検出row・フレーム・boxを照合して保存特徴をraw trackへ移す。予約3本はrun 18の `person_tracks` v5の実観測、ViTPose、CLIP特徴を使用する。校正は開発セットで#933、予約でrun 18のcourt receipt、向きは確認済みボール由来で固定する。予約 `video_001/clip_003` は#2444のcam0回復＋向きのreceiptを全条件で使う。このclipを除いた予約2本も併記する。3cam参照は既検出boxをレビューした部分参照で、予約3本のラベルには既採点の履歴がある。[参照の来歴](../../../src/tasks/player_association/review/README.md) [cam0回復](../run-multiview-pose-link-recall-20261124/holdout-camera-recovery.json)

- box採点は全raw IDの実観測を、camera・frame別にIoU 0.5で一対一照合する。選手のprecisionは採用された既知boxのうち選手の割合、非選手のprecisionは棄却された既知boxのうち非選手の割合。recallはそれぞれ全選手ラベル、全非選手ラベルを分母とする。rawに照合しなかった既知の非選手は棄却成功に入らない。曖昧と未照合の採用数を残す。
- 保持数とIDF1はcamera・frame・参照人物を単位とする既存採点器で得る。同一人物の重複ラベルは1単位へまとめ、未照合の選手単位をIDFNに残す。clip内の選手を50%以上保持した人数、raw IDとgroup IDのIDF1・switch・fragmentを集計する。
- camera間pairでは、MILP尺度Aの出力IDを選別グループの元rowへ戻す。全raw観測と照合できた参照unitを全条件で共有し、停止clipでIDを割り当てられなかった正解pairはFNに含める。split内でTP・FP・FNを合計してF1を求める。

4camは#2363のダブルス69本。保存済み `person_tracks` v7と校正から、player-based court side、人物対応、camera alignment、COCO17三角測量を再生する。CLIPは候補ごとに既存のサンプリング規則で保存済みrowの特徴を読む。4条件とも下流のアルゴリズムと閾値は共通。同じclip内で整列済みカメラと全関節入力（座標・信頼度・可視mask）が一致した場合は計算結果を共有する。完了clipは有効なCOCO17関節を1点以上持つclip。3D被覆は全69本の4選手枠×元フレームと、その17関節を共通分母とする。4人同時有効は4選手それぞれに有効関節が1点以上ある元フレームを数える。完了は人物の三角測量まで。4camの選手IDの正解と絶対3D位置は参照がなく、被覆の指標を用いる。[#2363の保存実行](../run-i2363-trt-player-video/README.md)

## 再現

checkout rootから起動する。E側の歴史的archiveを `MEIJI3_ARCHIVED_OUTPUTS`、#2363の保存runを `MEIJI4_I2363_RUN`、3cam datasetを `MEIJI3_DATASET` で指定できる。既定の場所はスクリプトを参照。CPUプロセス間の並列数は `--jobs`、BLASその他のスレッド数は各1。再実行は新しい出力ディレクトリを指定する。2D追跡の既存実験表は `history.py` から単独生成できる。

```bash
study=knowledge/runs/run-i2634-player-selection-ablation
bash "$study/reproduce.sh" outputs/i2634/repeat --jobs 4
```
