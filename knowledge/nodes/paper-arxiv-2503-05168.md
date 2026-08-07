---
{
  "id": "paper-arxiv-2503-05168",
  "type": "paper",
  "title": "Seele: A Unified Acceleration Framework for Real-Time Gaussian Splatting on Mobile Devices",
  "curator": "chatgpt-schedule",
  "date": "2026-08-07",
  "status": "reviewed",
  "external_ids": {
    "doi": null,
    "arxiv": "2503.05168",
    "openreview": null
  },
  "reviewed_at": "2026-08-08",
  "evidence_level": "fulltext-code",
  "tasks": [
    "synthetic_data_generation",
    "tennis_scene",
    "cross_cutting"
  ],
  "repo_paths": [
    "src/synthetic_data_generation/dataset/plcs/rendering/nht.py",
    "src/synthetic_data_generation/dataset/blcs/rendering/nht.py",
    "src/synthetic_data_generation/dataset/court/rendering/nht.py",
    "src/synthetic_data_generation/rendering/nht/composition_smoke.py"
  ],
  "sources": [
    {
      "kind": "publisher",
      "url": "https://openaccess.thecvf.com/content/CVPR2026/html/Zhu_Seele_A_Unified_Acceleration_Framework_for_Real-Time_Gaussian_Splatting_on_CVPR_2026_paper.html"
    },
    {
      "kind": "paper",
      "url": "https://arxiv.org/abs/2503.05168"
    },
    {
      "kind": "code",
      "url": "https://github.com/SJTU-MVCLab/SeeLe"
    },
    {
      "kind": "project",
      "url": "https://seele-project.netlify.app/"
    }
  ],
  "relations": [],
  "tags": [
    "literature",
    "synthetic-data-generation",
    "gaussian-splatting",
    "rendering-optimization"
  ]
}
---

## 要約

Seeleは3D Gaussian Splattingの前処理とラスタライズを同時に最適化する高速化手法である。Hybrid Preprocessingはcamera poseをクラスタ化し、各clusterで高寄与Gaussianをshared/exclusive集合へ分け、実行時に必要なsubsetだけをprefetchする。Contribution-Aware Rasterizationはopacity寄与の小さいGaussian-pixel workを早期に除外し、warp内の不要blendを減らす。

## 主要な主張と根拠

著者はCVPR 2026版で複数の3DGS系手法・データセットを評価し、3DGS適用時に平均約3.2倍の高速化とruntime Gaussian memoryの39.1%削減を報告している。公式CVF版でtitle、著者、venueを再確認した。CVF版の著者順はHe Zhu、Xiaotong Huang、Zihan Liu、Weikai Lin、Xiaohong Liu、Zhezhi He、Jingwen Leng、Minyi Guo、Yu Fengであり、初期candidateに混在していたarXiv版の著者順を正式出版版へ訂正する。

## tennis-labへの適用可能性

tennis-labのNHT rendererは48次元feature、expected depth、instance AOVを含むため、SeeLeのCUDA rasterizerをそのまま置換するのは契約違反になり得る。一方、静的court/background Gaussianだけをview cluster依存のactive-index集合へ絞り、既存gsplat呼び出しへ渡す前処理なら、NHT appearance space、instance ID、shader契約を維持したまま独立検証できる。人物Gaussianは常に全量保持する。

## 制約・失敗条件

原論文の主評価は標準3DGSのSH rasterizerとAGX Orin/Orin NX/A6000を中心とし、NHT拡張およびRTX 5060 Tiで同じspeedupを保証しない。view-dependent pruningを動的avatarへ適用するとvisibility holeを生む可能性がある。renderer-only導入では、追加fine-tuning由来の画質改善を期待値へ含めない。

## コード・データ・ライセンス

公式repositoryは公開されておりtop-levelの利用条件と、rasterization submoduleおよび上流Gaussian-Splatting由来コードの利用条件を分離して監査する必要がある。特にkernelコードの直接移植は避け、初回検証はtennis-lab側で独立実装したbackground active-index prefilterに限定する。外部datasetの取り込みは不要である。
