---
id: group-b00-clay-flare-images-steps-v1
type: group
task: synthetic_data_generation
sequence: 21
recorded_at: '2026-09-20'
title: 'B00 Flare: 50/100枚・7k/30kステップ比較'
artifacts:
  output_dir: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/comparison-flare-50-100-30k-v001
  comparison: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/comparison-flare-50-100-30k-v001/index.html
  metrics: /home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/comparison-flare-50-100-30k-v001/comparison.json
members:
- run-b00-clay-flare-nht-7k-v1
- run-b00-clay-flare-50-nht-30k-v1
- run-b00-clay-flare-100-nht-30k-v1
parents: []
papers: []
tags:
- synthetic-data
- nht
- clay
---

## 比較

同じ固定参照・Flare・プロンプトから作った画像群を用い、元SfMとカメラ、seed 42、factor 2、最大100万Gaussianを共有した。100枚版は既存50枚をそのまま含む。50枚版の全8評価視点を共通集合として、評価入力の画素ハッシュ一致を確認した。

| 画像数 | steps | 共通8枚 PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---:|---:|---:|---:|---:|
| 50 | 7000 | 22.7179 | 0.6527 | 0.2625 |
| 50 | 30000 | 23.1349 | 0.6247 | 0.2303 |
| 100 | 30000 | 24.2477 | 0.6588 | 0.1809 |

100枚30kが共通集合の3指標で最良だった。50枚を30kに延ばすだけではSSIMが低下し、白線の薄れも見られた。100枚版は白線を再現しやすくなったが、近いネットのぼけや線の欠けは残る。元画像の同条件対照は未実施であり、生成AIと元再構成の寄与は分離できていない。詳細・限界は各runの考察と比較HTMLを参照する。
