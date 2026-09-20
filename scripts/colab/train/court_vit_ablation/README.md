# Court DINOv3 size ablation on Colab L4

2026-09-20の[事前検証](../../../../knowledge/nodes/run-court-vit-l4-preflight-20260920.md)で、
既存Drive入力が別モデルのcheckpointと判明した。ユーザー承認により、論文用の
ローカルB（SHA-256: `b863df1f01f00d2ff00a21d56a461879e3c5af193a9f32cec47a88d2842f8383`）を
別名でDriveへ追加し、job manifestの再開入力を切り替えた。旧ファイルは上書きしない。

実行入口は `scripts/colab/train/court_vit_ablation.sh`。Colab/Driveの認証、入力stage、
固定git commit取得、GPU作成、進捗取得、終了処理は [共通workflow](../../README.md) が所有する。
ローカルGPUでは実行しない。

## 比較条件

`baseline.yaml` は `multiscale-depth3-local-rtx/config.yaml` を固定した設定。
Bはepoch 17/global step 29844の完全checkpointから、optimizer・schedulerを含めて
20 epochsまで再開する（0始まりのepoch 18、19）。S/S+/Lはseed 42で新規開始する。
全サイズでsynthetic 4 + TCD 4、BF16、学習解像度256/320/384/448/512、検証512、
同じ損失、20 epochsを使用する。DINOv3はfrozen。Transformer depth 8 / FFN 2048、
DPT large / 512 channels、dense head depth 3を固定する。

| サイズ | DINO幅 | アダプター | DINO出力層（0始まり） |
|---|---:|---|---|
| S | 384 | 4段それぞれ384→768 | 2,5,8,11 |
| S+ | 384 | 4段それぞれ384→768 | 2,5,8,11 |
| B | 768 | なし（既存checkpointと同一） | 2,5,8,11 |
| L | 1024 | 4段それぞれ1024→768 | 5,11,17,23 |

ユーザー指定に従い、S/S+/Lに独立した4本の学習可能な1×1 Convを追加する。
DPTへの全4段とTransformerへの最深段を768へ変換し、Transformer（12 heads）、
pose head、DPT、dense headsの構造・パラメータ形状をBと同じに固定する。
アダプターはfrozen DINOのno_grad境界の外で学習する。
Bには追加モジュール・パラメータを導入せず、完全状態resume互換性を維持する。

Colabでは全サイズでcompileを無効化し、コンパイル時間・一時メモリを避ける。
20 epochsを完走するためearly stoppingも無効化する。学習の損失やLRスケジュールは
変更しない。Bの最初の18 epochsは既存環境で学習済みなので、全20 epochsを同じL4で
実行した時間比較には使用しない。

## 起動・監視

repositoryの `.venv/bin/python` を使い、専用worktreeから実行する。

```bash
.venv/bin/python -m scripts.colab.train.court_vit_ablation.launch \
  --run-id court-vit-l4-20260920
bash scripts/colab/run.sh progress court-vit-l4-20260920
bash scripts/colab/run.sh logs court-vit-l4-20260920 --tail 40
```

6個のarchiveをDrive manifestおよび固定SHA-256と照合して一度だけ展開する。
古い `SHA256SUMS.txt` は現行court archiveと一致しないため使用しない。
展開後はB→S→S+→Lを同じL4で逐次学習する。各モデルは学習前に全Syntheticの
pose・教師データを検証する。明示的な`--smoke`は512px・batch 8の
`fast_dev_run`専用であり、本学習の前に4回の全件検証を追加する既定動作にはしない。
smokeは本学習checkpointを更新せず、productionのseed/optimizerを消費しない。
L4以外・容量不足・入力不一致・Drive同期失敗は停止する。

学習中のサイズ別出力は共通workflowの `colab-live/<run-id>/training/<size>/` に保存する。
`logs/version_*/checkpoints/last.ckpt` は各epoch終了時、`recovery.ckpt` は100 train steps
ごとに完全状態をatomic publishする。TensorBoard・configは60秒間隔で同期する。
切断直前の未保存ステップは失われる。エポック途中の再開は重み・optimizer・schedulerを
継続するが、DataLoaderのサンプル順・RNGの完全再現は保証しない。

## VM終了後の再開

同じIDを再利用せず、新しいrun IDと以前のIDを渡す。

```bash
.venv/bin/python -m scripts.colab.train.court_vit_ablation.launch \
  --run-id court-vit-l4-restart-01 --resume-from court-vit-l4-20260920
```

既存Drive出力を上書きしない。各サイズの `last.ckpt` / `recovery.ckpt` を新VMへstageし、
読み込んだglobal stepが最大の完全状態を採用する。採用した状態を新しい出力の
`<size>/resume/last.ckpt`へ先に保存するため、完了済みサイズも次々回の再開へ引き継がれる。未着手サイズは初期重みから開始し、
Bの元checkpointも入力として保持する。候補が一つもない場合や同名候補が重複する場合は
エラーにする。共通 `run.sh resume` はVM上のrequest再試行であり、このcheckpoint継続とは異なる。

## 検証

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -n0 \
  tests/e2e/colab/test_court_vit_ablation.py \
  tests/unit/tasks/court_detection/evaluation/test_configuration_multiscale.py \
  tests/unit/tasks/court_detection/data/processing/test_geometry.py
```

各scaleでisotropic resize、fx=fy、patch alignment、projection round-tripを確認する。
完全状態不足・学習条件変更のresume拒否、archive path traversal拒否、Drive入力と
サイズ別出力の分離も検証する。
