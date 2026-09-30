# 次のlane-A提案（実行・default変更なし）

## (a) 「poseなし」arm: court-only cache＋pilot

**推奨: court専用の厳密な入力契約をCPUで実装してから、2–4時間のcache生成と10–20分のpilot/評価を別枠で申請する。**
これは文脈なしpilotとの比較であり、full / 文脈のみ / pose摂動の完了を意味しない。
#964のperson契約・COCO全画面0.30既定の完了を待つ間、person/poseを一切生成・ロードしない。

- 入力はepoch9 evidenceと同じ329 train/val clip（TrackNet83 / Meiji108 / chat138、145,767 frame）。
  各clipの**frame 0だけ**のJPEGへ現在のcourt推論・領域探索を実行し、静的KP14を保存する。
  court checkpoint `b863df1f01f00d2ff00a21d56a461879e3c5af193a9f32cec47a88d2842f8383`と設定を固定し、
  store/evidence/JPEG/frame/PTS/source座標を束縛する。旧pose入りcacheは使用・削除しない。
- 現行`StoredJPEGContextProducer`は人物/pose生成が必須、`PilotConfig`とwindow/evaluation loaderも
  detector-onlyを要求するため、`use_court=true`の1行変更だけでは実行できない。
  court単独producer/cacheと、明示したcourt入力を学習・推論の共通経路へ渡す接続が必要。
  asset identityにDINO/ViTPoseを含めず、無効なposeを「生成済みで検出0」と表現しない。
  court未生成・破損・不一致は停止。探索実行済みの`no_supported_region`だけを診断付きall-falseとし、
  全clipを保持してcourt有効/無効別の件数・性能も報告する。領域探索fallbackを追加しない。
- 学習差分は`use_court=true`、新court cache、出力先だけ。epoch9 evidence・seed42・初期化・
  3source平衡・AdamW lr3e-4/WD.01・batch32・33frame/stride16・12epoch×250step・固定gapを維持。
  モデルには既存のzero-init court gateがある。同じ乱数/窓抽出を維持する検査を先に通す。
  refiner bestは同じMeiji選択18 clipのobserved/gap等重みNLLで選び、較正18 clipは選択に使わない。
- 比較は今回の文脈なしpilotと同一70 val clip/40,144 frame。保存済みdetector基準は再利用できる。
  現行`run_comparison`はrecipe一致を要求するため、courtの宣言された差分だけ許すablation入口を用意する。
  誤差p50/p90/p95、存在NLL、位置NLL、HDR50/90/95のcoverage＋面積を同じ層で比較し、
  camera群を保つpaired bootstrapを追加する。test video_001と較正fitは使わない。

| 段階 | wall見積もり | peak device VRAM見積もり | 新規disk見積もり | 根拠・上限 |
|---|---:|---:|---:|---|
| court-only cache 329 clip | **2–4時間** | **2–4 GB（未実測）** | **100 MB以内** | r10/r12のcourt stageは1clip12.60–38.26秒、6clip平均24.71秒。単純外挿135.5分。共有CPUの探索・起動・hash余裕を含む |
| court-only pilot＋r5診断＋paired GMM | **10–20分** | **2–4 GB（未実測）** | **200 MB以内** | 今回学習133秒、device3.01 GB。court token/gateとcompile・全val GMMを加算。dense detector再推論なし |
| CPU検証・hash回収・表/overlay | **5–15分**（実装時間を除く） | **0 GB** | **30 MB以内** | 全329 reader、同じデータ抽出/学習窓、court gate gradient、missing拒否を確認 |

6clipのcourt stageは`run-i935-context-fullframe-pilot-r10-20260928`の3件（30.29/12.60/20.12秒）と
`run-i935-context-shard-00059-tracknet-r12-20260928`（38.26秒）、
`run-i935-context-shard-00089-meiji-r12-20260928`（17.78秒）、
`run-i935-context-shard-00201-chat-r12-20260928`（29.19秒）が根拠。人物/pose時間は含めない。
court-onlyの実測VRAMは無く、上記は見積もり。モデル常駐による高速化も未測定なので織り込まない。

将来のgrant案: GPU排他1lane、allocator6 GiB/device-used7.5 GBで停止、絶対上限12 GB以内、RAM available≥6 GB。
court cacheは10clip程度のimmutable shard（約33件、各5–15分・timeout20分未満）に分け、
**最初の3clipの実測を回収してから**残りの予算を確定する。新worktree約1 GBを含め総disk1.5 GB以内。
これは申請案であり、今回はqueue登録していない。追加jobの実行枠・並列laneの割当はorchestratorが決める。

## (b) 既存refiner componentの既定assetをepoch9 detector＋新pilotへ変更する案

**今は変更せず、orchestratorが比較と動画をユーザーへ提示した後に判断する。**
今回の証拠だけでは、新組合せを全面的な精度・較正改善として推奨できない。

| asset | 現行の参照 | 提案する参照 |
|---|---|---|
| detector | ft-e13 `cd7927ad27e53ddd6aa77df28eca3c5e674552461ccda083a41e99e629857892` | mixed epoch9 `37f4c59aead00062829280ad591b3874886978891104a290f704a2c33c3c1b36` |
| refiner | r4 best epoch9 `ed659e142dd228949ec772bcfdf9db0f4c682081778853c15088f7202a848c76` | r18 best epoch10 `aec4ddbbd7f184adab52c019c176ebf252d91d1e7396ce80ee3cd576ea5aa6db` |

新pilotのinputは新detectorに束縛されているため、**2つを同時に切り替える**。
まず既存exporterで新しいimmutable bundleをCPUで作り、manifestにepoch9のhash/前処理/候補/窓規則を固定する。
本stackの`run_pipeline.py`は現在`--bundle`と`--detector-checkpoint`を必須指定し、標準
`src/tennis_scene/configs/pipeline.yaml`のdetector defaultはft-e13である。
したがって実際に変更する設定の正本と、既存componentを呼ぶrecipeのasset指定を揃える設定PRが必要。
専用recipeへの明示overrideによる検証と、ユーザーに提示する**default変更**を分けて記録する。
今回、bundle exportも設定書換えも行っていない。#936側のbranch/defaultには触れていない。

支持する証拠はMeiji val全cameraの裾誤差と候補recallの改善、既存GMM契約との互換性。
反証は新detectorより悪い中央値、TrackNet observedの退行、chat中央値の退行、較正側HDR95のcoverage不足。
詳しい分母・数値は[比較表](comparison.md)。Meiji観測正例の存在NLLだけでは不存在較正を保証できない。
入力は未較正の文脈なしGMMのままで、存在threshold・共分散補正・候補規則を同時変更しない。

変更を承認する場合の検証案:

1. CPUで新bundleをexportして全hash/strict loadを確認。
2. 既存専用componentを、固定のval clip_010の3camera×270frameで元動画からexecuteする。
   JPEG overlayは同じ評価入力の診断であり、元動画decodeの同等性検証の代わりにはならない。
3. 別プロセスのload-only、全GMM/存在・frame/PTS・artifact hashを検証する。
   旧r7の同clip/cam0ではdetector9.14秒＋refiner0.79秒を記録済みだが、新checkpointの元動画結果は未検証。
4. 共有`ball_detection`既定を変える範囲なら、その証拠を読むcourt_side等も新identityで再実行が必要。
   古いload-only artifactはhash不一致で停止させる。#936との接続・3D品質は別laneの受入であり、この結果だけで完了扱いしない。

| 段階 | wall見積もり | peak device VRAM見積もり | 新規disk見積もり |
|---|---:|---:|---:|
| CPU export＋strict load | 1–3分 | 0 GB | 10 MB以内 |
| 3camera/810frame execute＋load-only・CPU再監査 | **5–10分**（queue待ち除外） | **2–4 GB**、device7.5 GB停止案 | **150 MB以内** |
| default設定PRの通常CI | 10–20分 | 0 GB（CI runner） | ローカル追加ほぼなし |

既存重みを参照して複製しない。native heatmap72×128×810×float32は約29.9 MBで、GMM・候補・ログと余裕を加え150 MB。
上記は2componentの費用であり、標準scene全体・person/pose・三角測量の再実行費用は含めない。
これは実行許可やdefault採用決定ではなく、ユーザー提示用の変更内容・根拠・費用である。
