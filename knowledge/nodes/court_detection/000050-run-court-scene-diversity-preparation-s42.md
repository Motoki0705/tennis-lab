---
id: run-court-scene-diversity-preparation-s42
type: run
task: court_detection
sequence: 50
recorded_at: '2026-10-25'
title: Courtの固定6000枚・4シーン条件と合成／実画像valの学習準備
provider: codex
status: done
config: {backbone: dinov3_vits16, transformer_dim: 384, batch_size: 8, max_steps: 33160, train_images: 5500, synthetic_val_images: 500, real_val_images: 2211, training_seed: 42}
metrics: {gpu_probe_steps: 3, peak_allocated_bytes: 7230403584, peak_reserved_bytes: 7732199424}
artifacts:
  run_dir: knowledge/runs/run-court-scene-diversity-preparation-s42
  output_dir: outputs/court_detection/campaign/scene-diversity-6000-v1
parents: [group-i983-court-dinov3-shared1024-s42]
relations: []
papers: []
tags: []
---

## 考察 / Findings

既存のサイズ比較からDINOv3-Sと後段幅384を選び、B00のみ、B00+B01、B02まで、B03までの4条件を同じ6000枚・33160更新で比較する準備。2026-10-25のユーザー追加判断により、6000枚中500枚を合成valへ分離し、4scene条件は各125枚とした。trainは5500枚、実画像valは既存2211枚を前回と同じ順序で固定する。

画像生成はユーザー指定のGPT-6.1 Sol / xhighサブエージェントが別worktreeで行う。元train軌道を固定して撮影密度を上げ、train/valは軌道単位で分離する。親は完全な生成poolを検証した上で8個の選択manifestを読む明示source、元の実画像valだけを読むregistry、両domainを別々に評価する入口を実装した。subsetの重複・未知ID・同一軌道のsplit漏洩・checkpoint以降の入力変更は失敗させる。

CPU用の実際の生成fixtureで、固定更新数の学習とcheckpoint再読込、合成/実val別評価、順序変更の拒否を確認した。関連unit/integrationは712 passed。これは本番GPUの容量や実験精度を示すものではない。実データ生成後の4条件preflightが完了し、各scene枚数・nestedな選択・全条件をまたぐ軌道分離・元の実画像val順序と初期weightを照合した。RTX 5060 Tiで本番shape `[8,3,288,512]` の3更新が有限loss/勾配で完了し、peak allocatedは7.23 GB、reservedは7.73 GBだった。初回compile込み169.53秒、後続2stepは0.395/0.397秒。これは容量確認だけで、収束・精度やColab L4の速度の証拠ではない。本学習の精度は未測定。

その後ユーザーがDriveへのアップロードとColabでの4条件本学習を指示した。実行はサブエージェント、親の監視は30分周期とする。ローカル準備のplan/config/probeを履歴として保持し、Colab用のroot・Drive永続化・checkpoint全保持設定は新しい実行記録に保存する。

設定・再現コマンド・分割表・解釈上の制約は[準備bundle](../../runs/run-court-scene-diversity-preparation-s42/README.md)を参照する。既存の採用モデル判断は維持する。
