---
id: paper-2020-tracknetv2
type: paper
title: 'TrackNetV2: Efficient Shuttlecock Tracking Network'
year: 2020
authors:
- Nien-En Sun
- Yu‐Ching Lin
- Shao-Ping Chuang
- Tzu-Han Hsu
- Dung-Ru Yu
- Ho-Yi Chung
- Tsì-Uí İk
tasks:
- ball_detection
source: https://doi.org/10.1109/icpai51961.2020.00023
summary: バドミントン動画からシャトルを検出するU-Netベースのモデル。連続3フレームをチャネル方向に結合して入力し、対応する3枚のヒートマップを出力するMIMO設計を採り、focal lossで学習する。本文は取得できなかったため、この要約はWASB（arXiv 2311.05237v1）の付録Aの記述に基づく。
reason: '探索 search-wasb-comparison-depth2（起点 cite-baef928f0aefd5a6、指定役割 comparison、最大深さ
  2）から採録。

  引用元 Widely Applicable Strong Baseline for Sports Ball Detection and Tracking (cite-baef928f0aefd5a6)
  / ref-75; 役割 component, comparison, related, benchmark; 判定状態 classified; 根拠 Section
  1 (anchor:S1.p2.1.2, S1.I1.i1.p1.1, S1.I1.i2.p1.1); Section 2 (anchor:S2.p2.1);
  Section 3.1 (anchor:S3.SS1.p4.1); Section 4.1 (anchor:S4.SS1.p4.1); Table 1 (anchor:S4.T1);
  Section 4.2 (anchor:S4.SS2.p1.1, S4.SS2.p3.1); Table 2 (anchor:S4.T2); Section 5.2
  (anchor:S5.SS2.p1.1); Appendix A (anchor:A1.p5.1); References (anchor:bib.bibx75);
  判断 SBDT先行研究（TrackNetV2）として議論され、再実装してTable 2・Figure 6で性能比較されているほか、MIMO設計と入力設定（N=3,
  288×512）がWASBに採用され、Badmintonデータセットと分割方法も利用されている。

  Identification: OpenAlex global_search cursor_complete: unique_title_year_first_author_match

  引用元 MonoTrack: Shuttle trajectory reconstruction from monocular badminton video
  (cite-2f34272a6b8ccd3d) / ref-26; 役割 foundation, comparison, related, benchmark;
  判定状態 classified; 根拠 Section 2, anchor S2.p1.1; Section 3, anchor S3.p1.1; Section
  4.3, anchor S4.SS3.p1.1, S4.SS3.p2.1; bib [26] (anchor bib.bib26); 判断 §2でシャトル追跡の先行研究として挙げられ、§3ではTrackNetV2データセットを学習・評価データ（benchmark）として採用している。さらに§4.3では3-in-3-outのU-Net構成を出発点（foundation）とし、そのパラメータを初期エポックの知識蒸留にも用いる一方、パラメータ数（2.9M対11.3M）および精度（88.6%対84.0%）を直接比較（comparison）している。

  Identification: OpenAlex referenced_works: unique_title_year_first_author_match'
external_ids:
  openalex: W3122309393
  doi: 10.1109/icpai51961.2020.00023
---
