1. 判定・対象版

**CHANGES_REQUIRED**。評価1回目／指定1回。要修正はUIの状態更新漏れ1件。Generator／Discriminator、学習更新、GAN係数、ゼロノイズ、Flow／旧checkpoint互換性には今回の検査で要修正を認めなかった。

対象commit: `9588b33833c54423c361d150fbac442f9631497f`、base: `b8dab538e1121c4ccc357b31e4bf45a2e6f8ef1f`。開始・終了ともHEADとclean状態が一致し、指定30ファイルのSHA256も全件一致。manifestは `18fe4f79dcc1a030edc41922e61b15085a0de0bd25735c50a4a5e0ec8dd87cca` のまま。終了時に1280ラリーの内容hashをmanifestと照合し、カタログ内の互換checkpoint32件も開始時hashと一致した。

自身のrolloutのsession_meta／turn_contextから `openai / gpt-6-astra / effort=max / agent_path=/root/rope_gan_validator_1` を確認。親が提示した設定と一致する。同一モデル別コンテキストであり、見落としの統計的独立性は保証しない。

2. 原要件から組み立てた検査・自分で得た証拠

- アーキテクチャ・入力契約: Hydra合成後の設定を読み、2D／3DともG=幅256・8層、D=幅256・4層、FFN704、4heads・RoPE64・theta10000を確認。Gは共通TransformerBlock、DはBLCS／PLCSと同じbuilderで、入力は軌道だけ。Gの欠損値をNaNに差し替えても出力はbit単位で一致。実寸モデルをCPUでD更新→G更新し、detachによる分離、凍結DからGへの勾配、双方のパラメータ更新を検証。`coordinates/models/generators/transformer.py:30`、`coordinates/models/discriminators/__init__.py:16`、`coordinates/training.py:139`。
- schedule・loss・保存: 更新1／500／501／1000／1499／1500／1501／4000の係数は独立期待値 `[0,0,0.002,1,1.998,2,2,2]` と一致。CPU integrationで実runnerのD更新回数・復元loss＋重み付きGAN loss・2D／3D GAN／3D Flowの保存ロードと再評価を確認。実寸2D／3Dモデルも独自に保存ロードしbit単位一致。全欠損入力の長さ1／127／128／129／257で有限な全frame出力。`src/tasks/base/training/gan_schedule.py:6`、`coordinates/training.py:143`、同:183。
- データ・漏洩: 実datasetのtrain／val／test各1ラリーでevent率0／0.5／1を検査。2D／3Dが同じ2D拡張を使用し、欠損が選択イベント区間の和集合と一致。非欠損2DはGTと完全一致、noise／isolatedは完全な0。3Dの教師xyzを9999に差し替えても3D入力が変わらない反例を確認。
- legacy／Flow: 実v1の2D GAN・3D回帰・3D Flow checkpointを現実装とbase commitの元モデルにロードし、同じ入力／seedの出力がbit単位で一致。新FlowはCPU integrationの学習／保存／推論で確認。
- 通常検証: 自分で `.venv/bin/python -m pytest -n 0 -p no:cacheprovider --basetemp=/tmp/rope-gan-validator-1/pytest` を実行し **95 passed, 1 warning (9.17s)**。対象は `test_coordinate_rope_gan.py`、`test_coordinates.py`、`test_coordinate_review.py`、integration `test_coordinates_training.py`、baseの `test_gan_schedule.py`／`test_gan_transition_callback.py`／`test_gan_loss.py`／`test_gan_training.py`。`CUDA_VISIBLE_DEVICES=''`、`PYTHONDONTWRITEBYTECODE=1` と隔離cacheを使用。証拠は `/tmp/rope-gan-validator-1/pytest.log`。
- 独自CPU検証: `PYTHONPATH=. ... .venv/bin/python /tmp/rope-gan-validator-1/probes.py` が成功。詳細は `probes-summary.json`。queue登録script2本の `bash -n` も成功。GPUジョブの登録・実行はしていない。
- 実画面: `TMPDIR=/tmp/rope-gan-validator-1 node /tmp/rope-gan-validator-1/browser.cjs` でlocalhost:8786を操作。P95／jitter／outlier／isolated=0を受理し、rally_000030の462frame中、イベント3/9の46frameだけが同期欠損。新v2 2D＋旧v1 3Dおよび旧v1 2Dへの切替でCPU再推論に成功し、4camera×462frameの2Dと462frameの3Dを描画。pageerrorは0。詳細は `browser-summary.json`／`browser-requests.json`。
- 親の検証資料は上記独自検査後に読んだ。親報告の98 tests・ruff／mypy・GPU preflightの成功は自分の実行結果として数えていない。独自の画面操作で下記R1を追加検出した。

3. 重要度順の指摘

**R1 / P2（中）: 新設したjitter・外れ値率の編集だけでは入力／推論結果が更新・無効化されない。**

根拠: `src/tasks/ball_refiner/coordinates/review/static/app.mjs:270` のinputイベント登録に `noise-jitter` と `noise-outlier` がない。両値は `requestFromForm`（同:37）では読み取るが、編集時に `changedInput()` が呼ばれない。

再現: P95=200、jitter=3、外れ値率10%でCPU再推論後、jitterだけ4へ変更して800ms待つ。続いて外れ値率だけ20%へ変更して800ms待つ。どちらもpreview/inferリクエスト数は4のまま増えず、古い「CPU 推論」表示、RMSE、input hashが完全に残り、「設定を保存」も有効のままだった。どちらの変更値も有効な設定範囲内。

影響: 表示中の拡張設定と入力・予測・誤差が一致せず、新しい条件の結果と取り違えられる。ゼロノイズへの変更時も、他の項目を最後に編集するか再推論するまで新しい条件が適用されない。

改善案: 両IDを同じinput→changedInput経路に登録する。推論済み状態から各欄を単独編集し、古い予測と指標の消去・新値でpreview更新・その後の再推論を確認するブラウザ回帰検証を追加する。

スクリーンショットを自分で撮影し、`view_image`で目視確認した。

- `/tmp/rope-gan-validator-1/event-only-zero-controls.png`: 4項目0、実測P95=0.0、2D／3Dの同期した連続欠損、CPU推論とモデルstep表示を確認。新2Dは4更新のみのため軌道精度を合格扱いしない。
- `/tmp/rope-gan-validator-1/jitter-edited-stale-inference.png`: jitter欄が4に変わった一方で、旧条件のCPU推論・RMSE・入力hash表示が残るR1の状態を確認。

追加画像: `event-only-new2d-old3d.png`、`old2d-old3d-inference.png`。これらは保存済みだが、個別のview_image目視結果としては数えていない。

4. 未検証・制約・判定への影響

4,000更新の2D／3D本学習、GAN収束・位置精度、GPU上の独立再実行、compile有効時、全条件のブラウザ組合せは未検証。本学習は今回評価後に開始する合意なので、不足によるBLOCKEDにはしない。新3Dの実寸学習更新・保存ロードはCPUで確認したが、UIでの新3D checkpoint選択はカタログに対象がなく未実施。旧3Dと新2DのUI接続、および2D／3D共通loaderで補助確認した。

validator設定はread-onlyだが、このセッションの実効filesystem権限はunrestrictedであり、強制隔離とは主張しない。成果物／git状態／dataset／checkpoint／保存予測receiptを変更せず、自身の書込みを `/tmp/rope-gan-validator-1/` に限定した。修正・再委任・追加評価は行っていない。

版不一致や必須証拠の欠落はなく、確定したR1によりCHANGES_REQUIRED。評価対象外の正しさは保証しない。
