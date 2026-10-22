---
id: run-court-assets-colab-l4-20261022
type: run
task: synthetic_data_generation
sequence: 26
recorded_at: '2026-10-22'
title: B00〜B03のColab L4描画照合とGPU間の数値差
issue: 1351
provider: codex
date: '2026-10-22'
status: done
config:
  model: pinned_nht_3dgs_no_retraining
  data: B00_B01_B02_B03_observed_cameras
  batch_size: 1
  frames_per_scene: 4
  repeats_per_gpu: 2
  resolution:
  - 959
  - 539
  without_source_images: true
  asset_manifest_sha256: a9281d9a41e39fc944e684f7bdeb5b3517cb9424b297d318a557bcb263e8009b
  code_commit: 5d2ffbb692e68e0c0093e1358ba7fa6f7ab19f7d
  nht_commit: d4962c503f76d27fef27ac3f458edddc0a7d35c2
  setup_fix_commit: 7e8c926fe76c02d59c92b03c00e846297a2e2551
metrics:
  asset_file_count: 24
  asset_size_bytes: 958142735
  rendered_frames_per_gpu: 32
  kp_comparisons: 784
  renderer_visible_points: 498
  kp_disagreements: 0
  within_device_repeat_max_abs_error: 0
  equivalence_accepted: true
  l4_setup_corrected_warm_seconds: 898.111
  l4_peak_allocated_bytes: 491267072
  l4_peak_reserved_bytes: 658505728
  l4_sampled_device_peak_bytes: 943718400
artifacts:
  run_dir: knowledge/runs/run-court-assets-colab-l4-20261022
  raw_colab: gdrive:tennis_lab/outputs/synthetic_data_generation/profile/court-assets-verify/renders-native
  raw_local: gdrive:tennis_lab/outputs/synthetic_data_generation/profile/court-assets-verify/local-reference
  input_assets: gdrive:tennis_lab/data/synthetic_data_generation/court-scene-assets-v1-a9281d9a41e39fc9-verified
  staging_assets: gdrive:tennis_lab/staging/court-scene-assets-a9281d9a41e39fc9
  adoption: knowledge/runs/run-court-assets-colab-l4-20261022/adoption.json
parents: []
relations: []
papers: []
tags:
- court-detection
- colab
- rendering
- reproducibility
---

## 結論

資産転送・旧alignmentの読み取り・固定版NHTのL4描画を確認し、**2026-10-22のユーザー判断を出典として、教師一致とRGBのPSNR下限による等価性を受け入れた**。判断の原文は [#1351のコメント](https://github.com/Motoki0705/tennis-lab/issues/1351#issuecomment-6341401581)。全alpha/depth rasterの一致を合格に変更したわけではない。2回分のKP投影・renderer可視判定は完全一致し、RGBはREADMEに定義した下限を通過した。資産は入力データの採用位置へ検証付きでcopyし、stagingも保持する。後続の本番生成はColab L4・固定runtimeへ統一する。

これは完了した数値比較の記録であり、Court Detectionの精度実験ではない。同じGPUでの2回は全float32配列が完全一致したが、GPUを変えたときの差は2回とも再現した。GPU内のランダムな非決定性で説明できる証拠はない。

## 資産・コード・比較条件

資産の正本は添付の `assets.json`（`court_scene_assets_v1`）。B00〜B03の公開export・カメラ・疎点群・checkpoint・runtime config・確定alignmentをbyte単位で保持した24ファイル、958,142,735 bytesで、manifestを含む転送bundleは25ファイル、958,152,199 bytes。manifest SHA-256は `a9281d9a41e39fc944e684f7bdeb5b3517cb9424b297d318a557bcb263e8009b`。Drive配置・ローカル再取得・Colab取得後の全照合を通過した。stagingのpathとIDは `run.json` に記録した。

動画・captured images・SfM中間物・alignment heatmapは含めない。旧alignmentを既存の `load_alignment_result` とmetric adapterで読み、意味・幾何・座標変換を変更しない。この依存closureを完成済みscene-pipeline workspaceとして扱わない。

実行コードはtennis-lab `5d2ffbb692e68e0c0093e1358ba7fa6f7ab19f7d`、NHT `d4962c503f76d27fef27ac3f458edddc0a7d35c2`、gsplat `c8d5acd22314c1c1144a30e635e08d9c738a8b39`。Colabではinstallerのみ、ローカルの `7e8c926fe76c02d59c92b03c00e846297a2e2551` までの修正を適用した。回収patchのhunkはこのcommitと一致し、Git indexの短縮桁数だけが異なる。renderer・checkpoint・camera・alignmentの変更はない。コマンド・patch・各jobのsource snapshotを添付した。

4シーン各4つの既存公開カメラを、原解像度959×539、batch=1、appearance overrideなしで各GPU2回描いた。ローカルは共有training queueのresource=allを使用したRTX 5060 Ti、ColabはL4。Torch 2.9.1+cu130・gsplat 1.5.3・tinycudann 2.0・NumPy 2.4.6は一致し、Pythonはそれぞれ3.11.15/3.11.16。保存RGB/alpha/depthはfloat32である。内部FullyFusedMLPの演算を強制的にfloat32へ変更する実験ではない。

depthは `nht_scene_units_per_metre` で割ってメートルへ変換した。KP14の投影座標・class ID・physical index・in_front/in_frameとrenderer-visibleを比較した。renderer-visibleは既存のalpha>=0.01かつdepth>0の半径1px近傍判定であり、コート面との遮蔽距離一致を意味しない。既存packed datasetはJPEGと疎なラベルのみなので、alpha/depthの比較元は同じrequestのqueue再描画を使った。

## 数値差と許容差の判断

次の表は各シーン4視点を結合した全画素の値である。RGBは0〜1、depthはm。最大値も保持し、平均だけで合格にしない。

| Scene | RGB MAE / RMSE | PSNR dB | alpha MAE / 最大 | depth MAE / 最大 m |
|---|---:|---:|---:|---:|
| B00 | 7.78403e-05 / 0.000173169 | 75.231 | 1.44261e-08 / 0.00188786 | 2.33464e-06 / 0.166065 |
| B01 | 7.2212e-05 / 0.000211837 | 73.480 | 8.66891e-05 / 0.159166 | 31046.9 / 1.6215e+10 |
| B02 | 0.000349355 / 0.00126099 | 57.986 | 0.000300284 / 0.263484 | 0.143757 / 520.196 |
| B03 | 0.000101388 / 0.000161238 | 75.851 | 1.41208e-07 / 0.0102372 | 2.10363e-05 / 1.50378 |

392 KPの幾何・renderer判定は全点一致し、画像内かつrenderer-visibleは249点。投影UVとKP camera depthの差も0だった。2回分では784 KP、renderer-visible 498点を検査した。local/local、L4/L4のRGB・alpha・depth差は全画素0。local/L4の2回の比較結果は一致した。

シーン集計のRGB RMSEは0.000161〜0.001261で、同じローカル画像をquality=95のJPEGへencodeするRMSE 0.003537〜0.004983より小さい。ただし最大RGB差はB02の0.166748であり、JPEG平均誤差より小さいことだけをdepthや教師の等価性へ拡張しない。JPEG化は本番assemblerと同じ `np.round(rgb * 255).astype(np.uint8)` を使い、JPEG差をraw RGB差と分けた。シーン集計のJPEG RGBのGPU間PSNRは51.31〜54.24 dB。過去の異なる視点のJPEGとの比較ではない。

B01/frame_000088の(x=481,y=6)では、localのalpha=0.004035・depth=401.809mに対し、L4はalpha=0・depth=16,214,958,705.176mだった。この視点にはlocalで8画素、L4で10画素の「alpha=0かつpositive depth」があり、いずれも有効surfaceとは扱えない。gsplatのexpected depthがalphaを最小1e-10にclampして除算する実装と整合するが、除算前の量は今回保存しておらず原因の確定ではない。alpha=0の画素を除いても差は残り、両画像alpha>=0.99のB02でもdepth最大差12.124m。巨大値の除外だけでは全体の一致を説明できない。

B01のalpha=0.01閾値を跨ぐ画素は2点で、(505,6)は0.010164→0.009974、(482,12)は0.009825→0.010148。どちらも投影コート外で、今回のKP判定は変わらなかった。可視KP一致を全alpha raster一致として数えない。

コート領域は保存KPの四隅0,1,3,2をnear=0.01mとviewportでhomogeneous clippingし、PILでrasterizeした診断用mask。背景除去を受入条件へ後付けするためのmaskではない。全画素・両側alphaの閾値別・コート領域の分布と最大画素の値は `outlier-audit.json` に残した。

| Scene | コートpixel数 | RGB最大差 | depth P99 / P99.9 / 最大 m |
|---|---:|---:|---:|
| B00 | 603,958 | 0.013916 | 1.53624e-05 / 2.98714e-05 / 0.0610563 |
| B01 | 560,252 | 0.00341797 | 9.37818e-05 / 0.000985412 / 0.137215 |
| B02 | 505,829 | 0.0341797 | 0.000491203 / 0.00478158 / 0.0361511 |
| B03 | 366,027 | 0.000976562 | 9.40341e-06 / 4.1375e-05 / 0.0475549 |

同じGPU内の繰返し差が0なので、観測したGPU間差をそのまま「ノイズ床」として最大値以上の許容差へ広げる根拠はない。全rasterの最大差を許容するなら、背景の極大depthに加えてコート上の最大13.7cmも含まれる。コートdepthのP99.9は最大4.78mmだが、上位0.1%を削除したり値を補正したりしていない。ユーザー判断は教師一致を基準とする採用であり、これらのdepth値を一致・正確と認定するものではない。

診断の物差しとしてRGB 1/255、depth 1cmを使うと、コート領域で閾値を超える画素はB00が2/13、B01が0/57、B02が2951/114、B03が0/3（RGB/depth）。これらは原因を絞るための集計で、受入閾値ではない。alpha/depthの差、閾値跨ぎ、外れ画素を含む元の比較JSON・図・raw配列はすべて保持する。

### ユーザー判断に基づく採用基準とJPEG誤差の比較

再利用する契約の正本は [Court READMEのRendering equivalence contract](../../../src/synthetic_data_generation/dataset/court/README.md#rendering-equivalence-contract)。ユーザーが指定したのは「KP投影・renderer可視判定の完全一致＋RGBのPSNR下限」という判断軸であり、PSNRの数値自体は実測とJPEG保存誤差から具体化した。

シーン集計の最小PSNRはB02の57.985743 dB、個別視点の最小は同シーンframe_000104の55.858970 dBだった。「約58dB」を全視点の下限と取り違えない。4視点のRGB二乗誤差をまとめたシーン単位の下限を57 dB、平均に隠れる劣化を拒否する個別視点の下限を55 dBとし、実測最小値から整数dBへ切り下げた。同GPU内のノイズを推定して広げた許容差ではない。

quality=95保存の誤差は、raw RGBから本番と同じuint8丸め・JPEG encode/decodeまで含むRMSE 0.003537〜0.004983（PSNR 46.050958〜49.026667 dB）。57 dBの下限が許すRMSE 0.001413は、この最小JPEG RMSEの約0.399倍、55 dBの個別下限でも約0.503倍である。最悪視点の実測RMSE 0.001611は同じ視点のJPEG保存RMSE 0.004573の約0.352倍だった。JPEG量子化・圧縮より小さいRGB誤差予算として根拠を置くが、局所最大RGB差0.166748やalpha/depth差を消したり、学習精度への無影響を主張したりしない。

添付 `court_assets_acceptance.py` は元の比較JSON・両GPUの完全な教師inventoryを読み、2回を別々に判定した。`acceptance.json` に入力SHA-256、各シーンと最悪視点のPSNR、合否を保存し、2回とも合格した。微小なUV/depth・物理ID・可視判定の変化、RGB下限未達、欠落・重複視点、非有限値を拒否するCPUテストを追加した。従来の `run.json`・`verification.json`・測定sourceは当初の未受理状態を示す履歴として改変せず、現在の採用状態と判断の出典は `adoption.json` に追記する。

![各シーンでRGB最大差を持つ視点。Local/L4 RGB＋可視KP、RGB差×32、alpha差×4、対数scaleのdepth差](../../runs/run-court-assets-colab-l4-20261022/comparison.png)

## 速度・メモリ・導入時間

checkpoint loadは同期付きtorch.load、resident loadはshader構築まで、batch時間は同期待ち・CPU配列回収まで。先頭cameraを別記し、warm FPSは後続3cameraの合計時間で算出した。public callは入力検証・checkpoint hash・PNG/NPY保存を含む。数枚の既存cameraの測定であり、大規模生成のthroughputではない。

| Scene | L4 checkpoint load s | L4 warm FPS（2回の範囲） | L4 public FPS（2回、初回cold除外） | local warm FPS（2回の範囲） |
|---|---:|---:|---:|---:|
| B00 | 0.178–0.201 | 55.213–55.850 | 2.485–2.485 | 40.094–48.800 |
| B01 | 0.178–0.188 | 44.440–44.770 | 2.954–2.975 | 34.077–38.850 |
| B02 | 0.179–0.187 | 44.460–44.508 | 3.283–3.317 | 33.868–38.139 |
| B03 | 0.180–0.187 | 46.600–46.884 | 3.276–3.287 | 35.969–40.372 |

L4は最初のB00 batchが3.053sで、lazy compile等の初回処理を含む。L4の2回目B00は0.0258s、checkpoint loadは全シーン2回で0.1778〜0.2013s。PyTorch peak allocatedは459.6〜468.5MiB、reservedは620〜628MiB。100msサンプリングのdevice memory.usedはL4が892〜900MiB、localが2259〜2267MiB。device全体には背景使用が含まれ、観測peakは瞬間peakの下限である。正確な値と全batch時間は両GPUのmetrics JSONに置く。

Colabのuv syncは26.430s。最初のrender-only setupは951.585sでhelp/importを通ったが初回描画はABI不整合で失敗し、成功導入時間とは扱わない。修正後のsetupは既存cacheがある同VMで898.111sで、cold install時間ではない。Colabの資産download+verifyは49.987s、manifest/幾何検証は22.592s。最初のupload+verifyは135.532s、local download+verifyは115.386s。setup・転送・描画・Drive永続化の待ち時間を混同しない。

## 導入失敗と修正

最初はgsplat/csrc.soのundefined symbol `c10::cuda::CUDAErrorLogCapture` を観測し、続くJIT経路はNinjaをPATHから解決できず停止した。build isolation時のTorch未固定と整合するABI不一致で、build時Torchの正確な版は取得していない。installerを固定Torch導入後の同環境で `--no-build-isolation` によりgsplatだけを再buildする形へ修正した。再導入にもTorch/CUDA版を明示し、ninja等とtool-bin PATHを用意し、gsplat.csrcを直接importしてnative不整合を隠さない。広いreinstallでTorch更新が候補になった中間試行はcancel済みで、成功と数えない。

最初のNotebook提出はpendingのまま接続を失ってcancelし、保存を確認した。接続断では同じassignmentにreconnectし、実コマンドを重複投入しなかった。

## 中断からの回収・停止確認

前回agentのcapacity終了時にrender-nativeはfinalizingだったが、その後2026-10-22T07:38:48Zにcompleted/returncode=0/drive_saved_atをDriveへ保存済みだった。今回Driveから32枚分の全配列・manifest・source・metrics・visibilityを `download --verify` で回収し、元のmetricsと測定sourceの一致を確認した。7 job全てにterminal状態とdrive_saved_atがあり、失敗・cancelも保持した。setup-native-pinned、assets、render-nativeは成功。失われた必須成果物は見つからず、VM内の一時ファイル以外に再生成を要するものはない。

今回の `doctor --remote` はActive assignments=0、usage rate=0.00/hr。`stop` はSession is absent from Colab registryで拒否され、新VMを取得していない。今回のstop成功と偽らず、この結果を添付した。local baselineも別のDrive outputへverify付きで保存し、比較に使ったraw配列を残す。

## 再現と残課題

`run.json`、`render-inventory.json`、`colab-operations/`、`local-queue/` が実行版・コマンド・hash・状態の根拠。raw配列はDriveの `outputs/synthetic_data_generation/profile/court-assets-verify/{renders-native,local-reference}` に置き、Gitには含めない。取得した資産rootと両raw出力を指定し、添付 `recompute.sh` でCPU比較を再実行できる。学習していないためTensorBoard・収束曲線は対象外。

ユーザー判断後の採用先・Drive ID・照合結果は `adoption.json` とそこから参照するreceiptが正本。初回copy後のB01 `scene.json`同名重複と、別の新規先へのcopy中のDrive 500エラーを検出した。重複先を採用せず保持し、新規先のinventoryで1ファイルだけの欠落を確認してそのファイルだけをcopyした。最終inventoryは元と同じ25ファイル・958,152,199 bytesで、全path・サイズ・内容hashが一致し、重複なし。全量の再取得・内容照合と、元のSHA-256 manifest・全資産・幾何の再検証も通った。stagingも削除していない。

今回の仕上げではColab/GPUを起動せず、保存済みの各GPU32枚とCPU判定だけを使用した。関連テスト73件とNHT exportテスト46件が成功した。NHTテストの最初のcollectionは学習venvに`jsonschema`がなく失敗したが、既存NHT分離環境の依存を明示して成功し、環境の再installは行っていない。コマンド・ログは `adoption-checks.json` が参照する。外れ画素のrenderer/precision調査は今後の改善候補で、現在の採用条件では前提ではない。新しい軌道・品質gateの判断は後続へ残り、この既存撮影点の照合だけで新規視点を合格にしない。

軌道生成・大規模生成・学習・alignment再推定・wheel cacheは対象外。NHTのimage-free変更はoriginのbranchへpush済みだがmainには未mergeで、NHT側のPRが必要である。`nht-upstream.json` にremote refと包含確認を記録した。PRの作成とpushは親が行う。
