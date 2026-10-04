# Ball Refiner 2D dataset review

保存済みの入力証拠・人物/コート文脈・教師・GMMを、同じ実frameで調べる読み取り専用UI。
推論・生成・注釈編集は行わない。球付近の拡大、frame移動、教師maskのtimeline、状態別ジャンプで
「球の候補を見落としたのか」「教師が不明なのか」「文脈自体が未提供なのか」を分けて確認する。

## 起動

すべての入力pathは明示的な絶対path。Pythonはrepoの`.venv/bin/python`を使う。
worktreeからは元repoのデータ位置を指定する。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/kamimura/projects/tennis-lab/.venv/bin/python \
  -m src.tasks.ball_refiner.scripts.review_2d_dataset \
  --evidence /home/kamimura/projects/tennis-lab/data/ball_refiner/detector-mixed-e9-trainval-r17-20260930 \
  --predictions /home/kamimura/projects/tennis-lab/outputs/ball_refiner/evaluate/precision/i935-anchored-12k-s42-r21-20260930 \
  --context-root /home/kamimura/projects/tennis-lab/data/ball_refiner/context-fullframe-shards-r12-20260928 \
  --rgb-store /home/kamimura/projects/tennis-lab/data/ball_detection/ball-mix-v2 \
  --port 8894
```

`http://127.0.0.1:8894`へアクセスする。URLの`clip`・`frame`・`condition`で表示を再現できる。
`--predictions`・`--context-root`・`--rgb-store`は省略可能。未提供は明示される。
不完全なcache、破損hash、教師/PTS/座標の不整合はエラーになる。

## 現物の体系（2026-10-04確認）

| family / version | 役割・入力と教師・単位 | split / 現物 | 現行・過去の区別 |
|---|---|---|---|
| detector-mixed-e9-trainval-r17 | 候補8 slot、native score、5×5 patch、採用8frame窓。source `(W−1,H−1)` 正規化UV | train 259 clip / 105,623 f、val 70 clip / 40,144 f | 現行採用e9の保存証拠。参照するv1 storeは不在 |
| context-fullframe-shards-r12 | 全画面DINO→BoT-SORT→ViTPoseのモデル推定COCO17、frame 0のKP14。stored JPEG px | complete 119 clip shard（TrackNet 83、Meiji 35、chat 1） | 旧文脈比較。329 clip全体の完成cacheではない |
| precision / anchored-12k-s42-r21 | val教師理由・存在mask・GMM4成分・存在logit。教師は人手注釈由来、GMMはモデル推定 | 70 val clip × observed / evidence_gap、40,144 f / 条件 | 採用headの保存済み**未較正**比較評価。test・独立holdoutではない |
| calibration / anchored-s42-r23 | covariance倍率適用済み評価・bank材料 | 保存済み。今回はUI読込対象外 | 残っている較正成果物。bankのin-sample材料を独立評価にしない |
| Meiji frozen context r30 | 凍結人物経路の計画108 camera-clip、52,866 f | ローカルprogressはfailed、5 clip完了receipt。統合complete manifestなし | 後続の現行方針。途中生成物を完成cacheへ昇格しない |
| ball-mix-v2（上流store） | Meiji / TrackNet / chatのJPEG・注釈。元動画・人手観測、補間/推定を区別 | 1,300 clip / 814,657 f。train/val/test | 現行上流。ここでは明示的RGB提供元だけに使い、教師は旧NPZのまま |
| detector-ft-e13-r3、旧pilot等 | 同一v1 storeの旧detector / 絶対平均headなど | 現物あり | 比較・再現用。採用e9/headとの区別を保持 |

e9証拠の内訳はMeiji train 72 / val 36、TrackNet train 74 / val 9、chat train 113 / val 25 clip。
教師保存値の母数はobserved 31,167、interpolated 317、occlusion_estimated 121、
out_of_frame 269、no_instance 4,070、unresolved 4,200 frame。
今回の70 val clipにはunreviewed / multiple_instancesがなく、その挙動はunit testで検証する。
合成3D datasetは別の[Refiner 3D](../../refiner_3d/README.md)の対象。

現行/過去は、[taskの採用設計](../../README.md#採用設計-候補残差headanchored_12k)、
`configs/train.yaml`、保存manifestとGit履歴を突き合わせた。
`28ed88e8d`は旧全画面文脈の分割生成、`762e3e05f` / `85b63adf5`は凍結Meiji文脈、
`3dd8abdc6`は採用資産整理、`37c41ce42`はconfidenceフィルタの廃止。
旧文脈・絶対平均headは比較用として残る。物理的なv1不在をschema廃止と混同しない。

## 画面の読み方

- D1…D8は保存候補slot。scoreはdetector heatmap peakで、amodal存在確率ではない。
  dense heatmapはcacheに保存されていない。表示する5×5 patchの斜線は無効cell。
- 緑の＋はobserved教師。橙の◇は補間/遮蔽推定の参考位置で、位置mask・存在maskは0。
  明示的out_of_frameは位置mask=0、存在mask=1、存在教師=0。
  instanceなし・未レビュー・unresolved・複数instanceはunknownで、存在の負例にしない。
  教師NPZのないtrain clip等はunknownとは別の「未提供」。
- pose/courtは保存された**モデル推定**。全COCO17を表示し、入力用の肘・手首4点を強調する。
  raw peakを[0,1]へclipしてthreshold（既定0.5）を適用し、画像外の有限座標は保持する。
  frame 0のcourtを全frameで使う。実行済みで有効点なしと未提供を分ける。
- R1…R4は**同じ球**の位置仮説。存在を条件としたweightと、各成分のfull covariance由来の2σ楕円。
  楕円は混合HDRでも95%領域でもない。画面外にもGaussian tailはある。
  保存GMMは文脈なしで推論されたものなので、併置した旧contextをその入力だとは解釈しない。
- 人工evidence_gapは保存評価時の候補入力dropout。対象frameでは実効候補0件とし、
  原cache patchは無効と記した参考表示だけ残す。RGB遮蔽再推論ではない。

## 旧bindingとRGBの検証

e9 manifestは不在の`ball-mix-v1` metadata/index hashを保持する。
このUIは保存GMM manifestのinput hashと照合し、NPZ内の教師・frame/PTSを検証する。
上流v2の注釈やsplitを教師へ自動代入しない。

別指定RGB storeはclip ID、media hash、元/保存寸法、FPS/time_base、全frame index/PTS、
**実JPEG shard SHA256**が一致したclipのみ表示する。clip indexの一致だけでは採用しない。
UV→JPEGは `uv × (source_size−1) × clip.scale`（width比、両軸共通）。
不一致なら座標面で表示し、理由を出す。JPEGがないことをdatasetがないことに読み替えない。
contextも旧store hash・clip metadata・JPEG hash・frame/PTS・NPZ hashで証拠cacheへ束縛する。
e13由来contextをe9へ併置した事実は出自に明記する。

## 実装と検証

`artifacts.py`が版/identity/maskを検証、`web.py`はGET-only API、`static/`は描画・timeline操作。
既存の教師変換・候補契約・GMM契約・楕円変換を利用し、学習/生成/共有baseを変更しない。
保存cacheをレビューする新UIで、既存rendererのbefore画像は捏造しない。

```bash
.venv/bin/python -m pytest -n 2 --no-cov \
  tests/unit/tasks/ball_refiner/refiner_2d/test_dataset_review.py
PLAYWRIGHT_CORE_PATH=<installed-playwright-core-directory> \
  node tests/e2e/tasks/ball_refiner/dataset_review_2d_browser.cjs \
  http://127.0.0.1:8894 <capture-directory>
```

ブラウザcheckは上記実cacheで動くサーバーを対象とする。撮影素材は生成/ダウンロードしない。
Issue用スクリーンショットとmetadataは`assets/dataset_review/20261004/ball_refiner_2d/`に置く。
