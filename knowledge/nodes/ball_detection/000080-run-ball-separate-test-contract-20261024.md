---
id: run-ball-separate-test-contract-20261024
type: run
task: ball_detection
sequence: 80
recorded_at: '2026-11-12'
title: 別test配布版の固定評価契約と初回転送の検証
issue:
- 1476
- 1913
provider: claude
date: '2026-10-24'
status: failed
summary_impact: updated
config:
  lengths: [32, 64, 128]
  protocol: knowledge/runs/run-ball-all-frame-protocol-20261022/protocol.md
  test_payload_mode: external_source_view
  real_test_inference: false
  colab_used: false
metrics:
  test_clips: 225
  test_common_frames: 143074
  test_scored_frames: 127202
  source_files_stat_checked: 3336
  contract_verification_seconds: 1333.813361335997
  drive_partial_payload_files: 4
  drive_partial_payload_bytes: 136313568
repro:
  commit: '9043b045554d43ec03be7b0074f3031ae6190c65'
  command: .venv/bin/python knowledge/runs/run-ball-separate-test-contract-20261024/prepare_contract.py
artifacts:
  run_dir: knowledge/runs/run-ball-separate-test-contract-20261024
  output_dir: tmp/separate-test-contract-verified
  transfer_receipts: tmp/test-distribution-transfer
parents:
- run-ball-all-frame-protocol-20261022
- run-ball-portable-distribution-20261023
relations: []
papers: []
tags: []
---

## 結果と受入状況

別配布版のtestを学習側のidentityと区別して検証する固定contractを作成した。
合意済みprotocolに対し、225 clips、全T共通143,074 frames、教師あり127,202 framesが一致した。
T=32/64/128の評価窓はそれぞれ8,813/4,386/2,167で、各owner hashもprotocolと一致した。
詳細の正本は[local-verification.json](../../runs/run-ball-separate-test-contract-20261024/local-verification.json)。
元snapshotから両配布版のGT・PTS・geometry・教師・RGBを照合し、参照した3,336ファイルの
inode/size/mtime/ctime不変を確認した。RGBのcopy/hardlinkは0。全量のモデルtest推論とColab利用は0。

一方、test配布版のDrive配置は完了していない。
新規pathへの初回転送で`shards`が3重になり、driverが停止した。
全388ファイルの公開条件を満たさず、`distribution.json`も公開していないため、
このrun全体のstatusはfailedとする。精度・長文脈の仮説・採用モデルの判断は変更しない。

## 評価経路の設計

調査案Aに沿い、固定contract・学習run record・validation manifest・val reportを明示する。
対応表は両`distribution.json`を正本とし、contractはその内容hashと、clip対応・意味・各Tの
評価manifestのidentityを固定する。対応表を別の手編集台帳として複製しない。
学習のrun/checkpoint/model schemaとrecipeを維持し、重みのrecipe hashをrun recordへ、
recipe内の評価manifest hashをvalidationの保存bytesへ結び付ける。
test VMでtrain/validation payloadを再取得せずに照合でき、閾値・mode・混同行列はvalからそのまま適用する。

新しい評価はv2として出所と実装identityを保存する。旧v1は明示的に読めるが上書きしない。
比較動画のv2入口は明示したsnapshot rootと保存hashを検証する。
実装と通常検証の正本は、PR #1489の固定HEADにある[training README](https://github.com/Motoki0705/tennis-lab/blob/8b970dc090e86a9b0bde0fd2a6b26efe3d638a5f/src/tasks/ball_detection/training/README.md#trainvalidationとは別配布版のtest)と
本runの検証記録を参照する（このsrc変更はmain未統合。末尾の節を参照）。旧sourceで始めたrunのresumeには元source・環境・入力pathが必要である。

## 観測と測定の限界

初回の実manifest生成は完了したが、その後の実験scriptのprotocol key比較に誤りがあり停止した。
修正後は生成済みmanifestを明示し、新しい出力で全内容と対応を再検証した。
再検証は単調時計で約22.25分。`preparation_seconds=0`は再利用を意味し、
最初の準備を0秒と評価しない。最初の準備時間は欠測。UTC差分と単調時計の値は別の観測として保持する。
[execution.json](../../runs/run-ball-separate-test-contract-20261024/execution.json)と
[実行source patch](../../runs/run-ball-separate-test-contract-20261024/executed-source.patch)が、
データ照合時のsourceを固定する。157ファイルのdigestとの一致を確認してpatchを構成した。
後から加えたAPIの拒否検査を、この旧sourceで実施済みとは扱わない。
最終mapping guardでも実train/validation 1,537 clips・test 225 clipsの意味と対応が
同じcontractに一致することを確認した。
[current-contract-check.json](../../runs/run-ball-separate-test-contract-20261024/current-contract-check.json)は
この追加確認のmodule版を記録する。RGBの再hashは行わず、先の全内容照合と区別する。
[final-contract-check.json](../../runs/run-ball-separate-test-contract-20261024/final-contract-check.json)では、
配布file manifestだけを自己整合させた原注釈変更も受け入れない最終guardで、
全clipの注釈内容と元snapshot由来のannotation digestの一致を追加確認した。
[accepted-contract-check.json](../../runs/run-ball-separate-test-contract-20261024/accepted-contract-check.json)は
学習run recordと重みのsource hash/Torch版の照合を既存resume契約へ揃えた完成版での確認である。

## 通常検証と未解決の回帰

[validation.json](../../runs/run-ball-separate-test-contract-20261024/validation.json)のロック付き実行で628件が成功した。
別storeの実CLI 6条件、較正値の完全一致、対応・GT・PTS・RGB・原注釈・教師・ownerと
較正元の失敗注入、旧schema読込、両モデルのinline連続学習とresume、source変更拒否、動画、path/configuration監査を含む。
source内容とTorch版を保ったcheckout来歴の差を持つresume重みも、元run recordと照合して評価できた。
[synthetic-cli.json](../../runs/run-ball-separate-test-contract-20261024/synthetic-cli.json)と
[video.json](../../runs/run-ball-separate-test-contract-20261024/video.json)は24×24 CPU合成fixtureの証拠である。

既存の`num_workers=1`のhybrid resume回帰1件は、DataLoader待ちで停止するため明示的に分離した。
最初の総合実行は609件成功後に中断し、単独90秒・spawn明示90秒の診断でも停止した。
spawnの試験変更は戻した。基点の未変更コードでも45秒watchdogで同じ回帰が停止した。
原テストは保持し、学習runner/runtimeは修正していない。
[prefetch-regression.json](../../runs/run-ball-separate-test-contract-20261024/prefetch-regression.json)と
[baseline-source.json](../../runs/run-ball-separate-test-contract-20261024/baseline-source.json)を親へ渡す。
`num_workers=0`ではモデル・optimizer・schedule・RNG・抽出位置の厳密一致を両モデルで確認したが、
prefetch回帰の解消や全既存回帰の成功とは扱わない。

本runはデータ契約と機能の検証であり、実モデルのtest指標・GPU速度・VRAM・学習曲線は未測定。
TensorBoardは対象外で、合成fixtureのモデル出力を実データの精度として採用しない。

## Drive転送の停止と親への依頼

[transfer-failure.json](../../runs/run-ball-separate-test-contract-20261024/transfer-failure.json)に
新しいquota・宛先不存在の観測、作成先ID、3つの重複ID、全uploadの失敗、最終inventory、
supervisorのreturn code 1と両processの消滅を記録した。
4ファイル・約136.3 MBがinventoryに存在するが、配布版の完成とは認めない。
単調時計約245.5秒はsource hash・upload試行・失敗検査を含み、全量転送の速度や完了見積りに換算しない。

初回実行時のdriverは新規の`shards`を直列作成せず、最初の4 uploadを並列に開始した。
観測時刻と実装から、同時mkdirの競合が重複の原因と考えられる。
既存資産のtrash・移動・上書きは禁止されているため、重複検知後はDrive操作を停止した。
親が重複の修復または別pathへの配置を判断する必要がある。
再開前に新規subdirectoryを直列で準備するdriver修正と競合回帰を行い、
最後に全inventory・内容照合・terminal receiptを揃えることを推奨する。

2026-11-03にdriverを修正し、payloadの親directoryをworker起動前に直列作成したうえで、
inventoryの一意性と既存内容を照合するようにした。空の新規宛先への4並列転送と、
重複検知時にuploadを始めないことを含む回帰テスト8件が成功した。
同日の確認でも旧宛先には同名`shards`が3個残っており、元snapshotも記録されたローカルpathでは見つかっていない。
配布版の転送・inventory照合は未完了のままである。

## 2026-11-12の現状と記録の経緯

このノードはPR #1489のknowledge部分を、#1913でDriveとPRを再照合してmainへ記録したもの。本文の2026-10-24の観測と11/03のdriver修正の記述は当時のまま残す。

- **Driveの配置**: `data/ball_detection/ball-mix1800-720p-all-frame-test-v1` には同名の`shards`フォルダが3つ残り、payloadは計4ファイル・136,313,568 bytes（clip-00000〜00003）だけだった。3フォルダの中身は互いに異なり、同一内容の重複ではない。folder/file IDとsize、Drive metadataのMD5は[transfer-failure.json](../../runs/run-ball-separate-test-contract-20261024/transfer-failure.json)の記録と全7件で一致した。payload本体は取得しておらず、SHA-256の再計算はしていない。`distribution.json`・完了receipt・全Tの評価manifestは見つからない。
- **元snapshot**: 記録されたローカルの元snapshotと作業コピーは、今回の確認範囲では見つからなかった。配布を再開するには、元データを回収して内容照合できることが前提になる。3フォルダのどれかを正本と決めたり、残りを削除したりはしていない。
- **コード**: 評価契約・別storeのCLI・動画入口のsrc/tests変更と、`run-ball-portable-distribution-20261023` の転送driver修正（11/03）は、PR #1489をクローズした時点でmain未統合である。変更はbranch `codex/ball-temporal-split-evaluation`（固定HEAD `8b970dc0`）に残る。本ノードはその実装を検証した当時の記録であり、mainで同じ経路が使えることを示さない。
- **未完了**: 配布版の完成と全inventory照合、評価契約の最新mainへの統合、固定checkpointとvalidation較正による実test評価。いずれも#1476・#1367の残作業である。
