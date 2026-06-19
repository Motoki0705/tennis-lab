---
id: group-i524-dinov3-ssl-court
type: group
title: DINOv3 SSL → 下流 court detection (#524)
issue: 524
members:
  - run-i524-ssl-lora-vitb16
  - run-i524-convert-backbone
  - run-i524-court-seg-baseline
  - run-i524-court-seg-ssl
tags: [court-detection, dinov3, ssl]
---

## まとめ

テニスコート画像での DINOv3 LoRA SSL → バックボーン変換 → 凍結バックボーンでの court segmentation、
という一連のパイプライン。**SSL 済みバックボーンは凍結条件下で court seg の val mIoU を 0.517 → 0.800
に大幅改善**。ドメイン特化 SSL の有効性が確認でき、他下流タスクへの横展開が有望。
