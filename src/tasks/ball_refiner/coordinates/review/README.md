# 2D / 3D Refiner Dataset Review

共通の`single_object`を使い、GT・拡張後入力・学習済みモデルの全frame予測を
同じラリー／時刻で比較するローカルWeb UI。既存の共有Three.js描画を使う。
dataset・checkpoint・学習runは読み取り専用。

## 起動

座標Refiner実装を含むrepo / worktreeで実行する。

```bash
.venv/bin/python -m src.tasks.ball_refiner.scripts.serve_coordinate_review --port 8784
```

[http://127.0.0.1:8784](http://127.0.0.1:8784)を開く。
既定rootはGitの共通repoから解決するので、worktreeでも元repoのデータを使う。
別の配置は次の**絶対パス**引数で指定できる。

- `--data-root`: `data/ball_refiner/single_object`
- `--outputs-root`: `outputs/ball_refiner`
- `--checkpoints-root`: `ckpt/ball_refiner`（存在しなくても可）

CPU推論はサーバー内で実行する。GPUを選んだ場合は[共有training queue](../../../base/visualization/README.md)を通し、
workerでのみCUDAモデルをロードする。サーバーの既定はlocalhost限定。

## 比較操作

1. split・ラリー・cameraを選ぶ。4cameraの2Dと1本の3Dを同じframeで表示する。
2. 2Dと3Dのcheckpointを選ぶ。回帰／GAN／Flow、学習時イベント選択率、step、validation RMSEを表示する。
   `outputs`と`ckpt`配下を探索し、座標schema・次元・FPS・dataset manifestが一致した候補だけを選択可能にする。
   ★は同一評価レシピ内のvalidation最良。初期候補は既定の評価レシピから選ぶ。
   `last.ckpt`も表示できる。非互換の重みは理由を示し、GMMなどを読み替えない。
3. 緑=GT、灰=拡張後入力、橙=推論。重ね表示／横並びを切り替える。
   3DはGTとコートに合わせた視点から始まり、回転・拡大と視点リセットができる。
   入力の欠損をまたぐ線は描かず、推論は観測区間も含む全frameを表示する。
4. 再生・速度・コマ送り・イベント移動・timelineのクリックを使う。
   timelineはshot/bounce、選択cameraの連続／離散欠損、三角測量後の3D欠損を区別する。
5. 全体・欠損・観測・イベント±5frameの位置RMSE、速度RMSEを確認する。
   時系列は位置、速度、GTからの距離誤差。2Dは選択cameraのpx、3Dはm。
   入力の大きな外れ値でグラフの範囲が広がる場合は、入力の表示チェックを外すとGT／推論に合わせて拡大される。

## 拡張と推論

イベント選択率、前後幅の範囲、離散欠損率、ノイズP95、seedを変更できる。
左右幅は異なる値を抽選する。「欠損のみ」「ノイズのみ」「なし」も用意した。
学習と同じ拡張関数を使い、同じノイズ・欠損から2D入力と三角測量後の3D入力を作る。
実際のframe欠損率と、現在のラリーの非欠損frameで測ったノイズP95を表示する。
P95の入力値は分布の設定なので、短い1ラリーの実測値は一致するとは限らない。

入力条件やモデルを変更すると古い推論を消し、「再推論」で選択モデルを実行する。
Flow seedは拡張seedと独立で、平均や正解に基づく選択をせず1本だけ生成する。
再生frame・camera・表示方法の変更だけでは推論しない。

checkpoint内容hashと結びついた表示用test予測があり、ラリー・拡張条件・seedが一致する場合は
「保存済み評価」で再計算せず表示できる。GT、入力、mask、camera/frame対応も検証する。
条件不一致を現在のモデルの結果として表示せず、評価条件への復帰または再推論を求める。
checkpoint本体にない学習条件は推測せず、隣接configがないコピーでは条件不明と表示する。

生成元の重みhashがない旧学習runの予測は、そのまま流用しない。次のコマンドは互換性のある
`best.ckpt`からtest予測をCPUで新しく生成して終了する。学習runは変更せず、
`outputs/ball_refiner/review/predictions/<checkpoint hash>/<評価条件 hash>/`へ保存する。
同じ内容を検証済みなら省略し、破損したbundleは黙って上書きしない。

```bash
.venv/bin/python -m src.tasks.ball_refiner.scripts.serve_coordinate_review --prepare-saved-predictions
```

重み・dataset・評価条件・予測本体のhashを照合する。同じstepの重みを差し替えて一覧を更新した場合も
古い予測を流用せず、表示用予測の生成またはUIの再推論を要求する。

## 保存と再現

「設定を保存」はdataset / checkpoint hash、ラリー、拡張、seed、表示frame等をJSONとしてdownloadする。
「設定を読み込む」はhashを確認し、保存評価または同じseedの再推論で状態を復元する。
「比較画像を保存」は2D/3Dの現在の表示、timeline、グラフ、モデル・拡張条件をPNGにする。
ファイルの保存先はブラウザのdownload設定による。datasetや学習runへ書き戻さない。

## 検証

データ・API・モデル互換性・保存結果の同一性は
`tests/unit/tasks/ball_refiner/test_coordinate_review.py`で確認する。
GPU routingは既存の共有queue境界をテストする。

実データと学習済み重みを配置してサーバーを起動後、Playwrightで操作・スクリーンショットを検査できる。
`PLAYWRIGHT_MODULE`は環境にインストールしたPlaywright packageの絶対パスを指定する。
出力先はsourceと分離した任意のdirectoryを指定する。

```bash
PLAYWRIGHT_MODULE=/absolute/node_modules/playwright \
REFINER_REVIEW_ARTIFACTS=/absolute/review-evidence \
node tests/e2e/tasks/ball_refiner/coordinate_review_browser.cjs
```

ブラウザ検証は実際の保存予測・CPU回帰/Flow推論、拡張切替、横並び、orbit、
JSON/PNG出力と復元、遅延応答とシーン切替の競合、390/800/1600pxの配置を確認する。
