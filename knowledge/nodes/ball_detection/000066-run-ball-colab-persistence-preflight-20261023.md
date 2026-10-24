---
id: run-ball-colab-persistence-preflight-20261023
type: run
task: ball_detection
sequence: 66
recorded_at: '2026-10-23'
title: 配布storeの注釈path未解決によりColab保存再開の事前検査を停止
issue: 1375
provider: codex
date: '2026-10-23'
status: failed
config:
  window_length: 32
  max_window_length: 128
  input_mode: source_payloads
metrics:
  eligible_chat_clips: 987
  input_file_versions_checked: 2921
artifacts:
  run_dir: knowledge/runs/run-ball-colab-persistence-preflight-20261023
parents:
- run-ball-portable-distribution-20261023
- run-ball-temporal-heatmap-gpu-diagnostic-20261023
relations: []
papers: []
tags: []
---

Colab保存・新VM再開へ進む前のCPU検査で、配布storeと窓生成の接続不備を確認して停止した。
`status: failed`はこの事前検査の結果であり、GPU学習・Drive保存・新VM復旧の失敗を観測したものではない。
担当範囲の指定に従い、先行の窓実装を変更していない。#1375の受入条件は未達である。

[実データreceipt](../../runs/run-ball-colab-persistence-preflight-20261023/annotation-path-probe.json)は、
先行runと同じ1,537 clipの配布manifestを使い、RGBをコピー・hardlinkせず元snapshotを明示参照した。
Chatの987 clipが「確認済み空frameと128枚以上の共通区間を持つ」検査対象となる。
そのうちclip ID順の先頭で、配布版の`store.annotation_path(clip)`は原注釈を取得でき、hashも一致した。
元snapshotでは同じclipのreviewed/target/instance数が一致し、空frameの証拠検査に成功した。
これに対して配布版の`reviewed_empty_evidence`と`build_all_frame_manifests`は
`Chat reviewed-empty annotation missing or checksum differs from the snapshot`で停止した。
2,921入力ファイルについてinode/size/mtime/ctimeの不変も確認した。
全RGBのhash再検査・再転送・モデルの実行はこの診断では行っていない。

原因は2か所にある。`all_frame_preparation.py`はmetadataの相対注釈pathをそのまま
`file_version`へ渡すため、checkoutをcwdにすると`FileNotFoundError`となる。
`hybrid.py:reviewed_empty_evidence`は同じ値に絶対pathを要求しており、cwdを変えても通らない。
配布契約のresolverを両方の消費側で使い、既存のhash・frame・notes照合を維持する修正を推奨する。
manifestを元snapshot用から流用する、metadataを絶対pathへ書き換える、空frameを除外する回避は採らない。
既存の配布・窓テスト44件は成功したが、この組合せの検査が無く、単独の成功から統合成功は判断できない。

resumeの配置は`/content/tennis-lab`をVM間で共通にする方針とした。
store・生成manifest・runの具体的な固定配置は[再開手順](../../runs/run-ball-colab-persistence-preflight-20261023/README.md)に記した。
既存のpathを含むidentity照合を維持し、内容hashを理由にpath差を無視する入口は追加しない。
ただし注釈の相対pathは別の消費側不備であり、配置固定だけでは解決しない。
配置方針の新VMでの実証は停止後の後続作業に残る。

[runtime事前確認](../../runs/run-ball-colab-persistence-preflight-20261023/runtime-preflight.json)はCLI 0.7.4の認証・利用枠を確認した。
このタスクではsession割当・job提出・Drive書込みを行っていない。
アカウント全体では所有者未確認の既存割当1件が観測されたため、それを停止していない。
本タスクの所有sessionは0件だが、全体の割当0件・停止を検証したとは扱わない。
利用枠の差分は既存割当の活動を含むため、本タスクの消費量として帰属させない。
データ取得・環境構築・学習・保存・復元の時間と損失列は未測定で、TensorBoardも生成していない。

次は親が先行範囲のresolver接続を修正し、移動済み配布fixtureで`prepare_data --mode all_frames`と
notes付き空frameの除外を確認する。その版をこのbranchへ反映後、全配布store用manifestの再生成、
checkpoint直後のhash照合付き保存・失敗注入、720pの短いL4 runと新VM復旧を再開する。
これは従来の配布・CPU学習経路の証拠を否定するものではなく、両者を接続した際の未検証境界を示す。
精度・モデル優劣・採用checkpointの判断は変更しない。

2026-10-24に再開条件を確認した。親が先行修正をこのbranchへmergeし、両消費側がresolverを使う状態になった。
[先行の実データ照合](../../runs/run-ball-portable-distribution-20261023/consumer-verification.json)では、
6 manifestの共通793,017 frame・39,185 pack・owner・教師kindの一致と入力不変を確認済みである。
このcheckoutでも移動済み配布版の窓生成・dataset読込と注釈破損拒否を含む6テストが成功した。
停止時のreceiptは当時の証拠として維持し、Colab保存・新VM再開の作業を再開する。
既存割当の所有者は引き続き未確認であり、親の追加指示に従い今回作成するsessionだけを停止対象とする。

後続の[720p保存復旧診断](000067-run-ball-colab-checkpoint-resume-20261024.md)で、
保存失敗の伝播と、最初のVM停止後に作成した新VMでの同run再開・対照との照合・両sessionの停止を確認した。
このノードの失敗receiptは当時の事前検査として維持する。
