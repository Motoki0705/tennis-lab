# PLCS Web UI 利用ガイド

選手の3D位置・姿勢・軌跡を、コートとカメラ位置とともに確認します。
[共通の実行前確認](../../base/visualization/README.md#実行前確認)を済ませ、コードのあるリポジトリまたはworktreeの直下で実行してください。各ブロックは単独でコピーできます。

## データセット閲覧

```bash
ROOT="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
./scripts/run_in_repo_venv.sh python -m src.tasks.plcs.scripts.review_dataset \
  --data-root "$ROOT/data" \
  --port 8772
```

[閲覧UIを開く](http://127.0.0.1:8772)。左の形式・シーンを選択すると、checkpoint・GPUなしでGTが表示されます。対象はphysical_v1のsingle_objectです。

中央の3Dコートと右の「2D入力 ↔ 3D教師」が同じフレームに同期します。
右で保存カメラを選ぶと、COCO17とコートの保存2D観測を3D教師の再投影と重ねて確認できます。
全体図は正規化UVの画像領域、拡大図はvis=1の選手関節を示します。
コートは既定学習設定の先頭14点を表示し、保存20点へ切り替えられます。

- root位置[m]・yawは保存された教師です。中央HUDの`hip centre`はCOCO17の左右hip中心です。
- 選手の有効観測なし、可視性の不一致、vis=1の画面外、再投影差が0.5pxを超える点を確認できます。
- 元動作・動作種別・保存canonical関節数・sceneのsplit・同じ元動作のsplit共有を表示します。
- 必須2Dファイルの欠落や形状・有限値の不正はエラーとして表示し、3Dの表示から区別します。

対象はACCAD由来の合成投影で、RGB映像・実測3D教師はありません。
visは「カメラ前方かつ画像内」の判定です。人体・コートの遮蔽は表しません。
学習時のカメラ選択・時間窓crop・augmentation前の保存データを検品する画面です。

## 推論

```bash
ROOT="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
./scripts/run_in_repo_venv.sh python -m src.tasks.plcs.scripts.serve_inference_ui \
  --project-root "$ROOT" \
  --data-root "$ROOT/data/plcs" \
  --checkpoint-root "$ROOT/outputs/plcs" \
  --extra-checkpoint-root "$ROOT/ckpt/plcs" \
  --device cuda --port 8771
```

[推論UIを開く](http://127.0.0.1:8771)。閲覧と同時に使う場合は別ターミナルで起動します。

1. 左でcheckpointを検索・選択します。保存設定と互換な形式だけに絞られます。
2. 形式・シーンを選びます。checkpoint未選択でもGTは確認できます。
3. 右でdevice、使用カメラ、reference camera、開始フレーム・窓長、canonical poseの由来を指定します。
4. 推論を実行し、GTと予測を重ねて再生します。右で位置・yaw・関節誤差を確認します。

GPU要求は[共有キュー](../../base/visualization/README.md#web-uiのgpu実行)で実行します。CPUで試す場合は起動引数を`--device cpu`にするか、画面でCPUを選択します。

## 表示操作

ドラッグで回転、Shift＋ドラッグで移動、ホイールでズーム、ダブルクリックで視点を戻します。カメラ位置・視錐台・カメラ視点、追従・軌跡を切り替えられます。下部で再生・一時停止・フレーム送り・シーク・速度を調整します。

## パスと注意点

- 閲覧の`--data-root`は`data`、推論は**`data/plcs`**です。
- `--extra-checkpoint-root`は繰り返し指定できます。追加重みがない場合はその行を省略できます。
- 別の保存先を使う場合は各root引数を実際の絶対パスに置き換えます。
- ポートが使用中なら`--port`を空き番号へ変更し、その番号のURLを開きます。終了はCtrl+Cです。
- 生のACCADモーションは[ACCAD review](review/README.md#accad-motion-review)で確認します。

[モデル対応・推論契約](inference/README.md) / [閲覧API・座標系](../../base/visualization/review/README.md)
