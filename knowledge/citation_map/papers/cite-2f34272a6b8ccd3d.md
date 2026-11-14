---
id: cite-2f34272a6b8ccd3d
type: citation_paper
title: 'MonoTrack: Shuttle trajectory reconstruction from monocular badminton video'
identity_key: doi:10.1109/cvprw56347.2022.00395
year: 2022
authors:
- Paul Liu
- Jui-Hsien Wang
external_ids:
  openalex: W4225510413
  doi: 10.1109/cvprw56347.2022.00395
  arxiv: '2204.01899'
source: https://arxiv.org/html/2204.01899
summary: 本研究は、単眼のバドミントン試合動画からコート検出・選手姿勢推定・シャトル2D追跡を統合し、人手介入なしでショット分割と3Dシャトル軌道再構成を行う初のエンドツーエンドシステム「MonoTrack」を提案する。グラフベースのコート検出、残差接続と重み付きDice＋BCE損失を用いた軽量U-Netによるシャトル検出、GRUベースのヒット検出ネットワーク（HitNet）と制約付き動的計画法によるショット分割を組み合わせ、空気抵抗モデルに基づく非線形最適化で3D軌道を推定する。拡張TrackNetV2データセットおよび10kの合成軌道データセットでの評価により、コート検出、2Dシャトル追跡、ヒット認識で従来手法を上回る性能を示し、3D軌道再構成の有効性と誤差要因を明らかにした。
text_status: available
text_reason: ''
references:
- ref_id: ref-1
  citation: '[1] Yousset I Abdel-Aziz, HM Karara, and Michael Hauck. Direct linear
    transformation from comparator coordinates into object space coordinates in close-range
    photogrammetry. Photogrammetric engineering & remote sensing, 81(2):103–107, 2015.'
  target_id: cite-ee694b10d11ddffa
  identification_status: resolved
  roles:
  - component
  decision_status: classified
  reason: '3D軌道再構成において、推定した3D軌道を画像空間の2D軌道へ投影しカメラパラメータを得るためのDirect Linear Transform（DLT）手法として採用されている。

    Identification: unique exact title/year match in OpenAlex'
  evidence:
  - Section 4.5 (Estimating the initial conditions), anchor S4.SS5.SSS0.Px2.p1.1
  - bib [1] (anchor bib.bib1)
- ref_id: ref-2
  citation: '[2] Hua-Tsung Chen, Wen-Jiin Tsai, Suh-Yin Lee, and Jen-Yu Yu. Ball tracking
    and 3d trajectory approximation with applications to tactics analysis from single-camera
    volleyball sequences. Multim. Tools Appl., 60(3):641–667, 2012.'
  target_id: null
  identification_status: unresolved
  roles:
  - related
  decision_status: classified
  reason: 'コート検出（§4.1）において、Farinらのモデルベースコート検出アルゴリズムに関連するスポーツ映像での同種先行研究の1つとして引用されている（具体的なベースライン比較対象は[11]、実装基盤は[4]）。

    Identification: OpenAlex referenced_works (33 candidates): year_mismatch_or_missing'
  evidence:
  - Section 4.1, anchor S4.SS1.p1.1
  - bib [2] (anchor bib.bib2)
  lookup:
    title: Ball tracking and 3d trajectory approximation with applications to tactics
      analysis from single-camera volleyball sequences
    year: 2012
    authors:
    - Hua-Tsung Chen
    - Wen-Jiin Tsai
    - Suh-Yin Lee
    - Jen-Yu Yu
    links: []
- ref_id: ref-3
  citation: '[3] Hua-Tsung Chen, Ming-Chun Tien, Yi-Wen Chen, Wen-Jiin Tsai, and Suh-Yin
    Lee. Physics-based ball tracking and 3D trajectory reconstruction with applications
    to shooting location estimation in basketball video. Journal of Visual Communication
    and Image Representation, 20(3):204–216, Apr. 2009.'
  target_id: null
  identification_status: unresolved
  roles:
  - related
  decision_status: classified
  reason: 'バスケットボール映像における物理モデルに基づく3D軌道再構成とショット境界検出の先行研究として§1・§2で言及され、§4.1でもFarinらのモデルベースコート検出法に関連する先行研究群の1つとして引用されている。

    Identification: OpenAlex referenced_works (33 candidates): year_mismatch_or_missing'
  evidence:
  - Section 1, anchor S1.p3.1
  - Section 2, anchor S2.p2.1
  - Section 4.1, anchor S4.SS1.p1.1
  - bib [3] (anchor bib.bib3)
  lookup:
    title: Physics-based ball tracking and 3D trajectory reconstruction with applications
      to shooting location estimation in basketball video
    year: 2009
    authors:
    - Hua-Tsung Chen
    - Ming-Chun Tien
    - Yi-Wen Chen
    - Wen-Jiin Tsai
    - Suh-Yin Lee
    links: []
- ref_id: ref-4
  citation: '[4] Grzegorz Chlebus and Thomas Sablik. Fully automatic algorithm for
    tennis court line detection. https://github.com/gchlebus/tennis-court-detection,
    2019.'
  target_id: cite-a51068d63615c196
  identification_status: resolved
  roles:
  - foundation
  decision_status: classified
  reason: 'コート検出モジュール（§4.1）の実装を同文献が提供するリファレンスコードの上に構築しており、コート検出の実装基盤として採用されている。

    Identification: explicit external ID in reference'
  evidence:
  - Section 4.1, anchor S4.SS1.p4.1
  - bib [4] (anchor bib.bib4)
  lookup:
    title: Fully automatic algorithm for tennis court line detection
    year: 2019
    authors:
    - Grzegorz Chlebus
    - Thomas Sablik
    links:
    - https://github.com/gchlebus/tennis-court-detection
- ref_id: ref-5
  citation: '[5] Xiangtong Chu, Xiao Xie, Shuainan Ye, Haolin Lu, Hongguang Xiao,
    Zeqing Yuan, Zhutian Chen, Hui Zhang, and Yingcai Wu. TIVEE: Visual Exploration
    and Explanation of Badminton Tactics in Immersive Visualizations. IEEE Transactions
    on Visualization and Computer Graphics, pages 1–1, 2021. Conference Name: IEEE
    Transactions on Visualization and Computer Graphics.'
  target_id: cite-eb2f0ee06381e51b
  identification_status: resolved
  roles:
  - related
  decision_status: classified
  reason: 'バドミントンの3D軌道を用いた戦術分析・可視化システムの先行研究（confirming pointsによりショット単位で人手介入を要する手法）として§1および§2で言及されている。

    Identification: OpenAlex referenced_works: unique_title_year_first_author_match'
  evidence:
  - Section 1, anchor S1.p2.1
  - Section 2, anchor S2.p1.1, S2.p2.1
  - bib [5] (anchor bib.bib5)
  lookup:
    title: 'TIVEE: Visual Exploration and Explanation of Badminton Tactics in Immersive
      Visualizations'
    year: 2021
    authors:
    - Xiangtong Chu
    - Xiao Xie
    - Shuainan Ye
    - Haolin Lu
    - Hongguang Xiao
    - Zeqing Yuan
    - Zhutian Chen
    - Hui Zhang
    - Yingcai Wu
    links: []
- ref_id: ref-6
  citation: '[6] Anthony Cioppa, Adrien Deliege, Silvio Giancola, Bernard Ghanem,
    Marc Van Droogenbroeck, Rikke Gade, and Thomas B. Moeslund. A context-aware loss
    function for action spotting in soccer videos. In IEEE/CVF Conference on Computer
    Vision and Pattern Recognition (CVPR), pages 13123–13133. IEEE, 2020.'
  target_id: cite-ef3765ca64d3dee3
  identification_status: resolved
  roles:
  - related
  decision_status: classified
  reason: 'スポーツ映像における行動の局在化・分類（activity localization and classification）の先行研究として§2で言及されている。

    Identification: OpenAlex referenced_works: unique_title_year_first_author_match'
  evidence:
  - Section 2, anchor S2.p1.1
  - bib [6] (anchor bib.bib6)
  lookup:
    title: A context-aware loss function for action spotting in soccer videos
    year: 2020
    authors:
    - Anthony Cioppa
    - Adrien Deliege
    - Silvio Giancola
    - Bernard Ghanem
    - Marc Van Droogenbroeck
    - Rikke Gade
    - Thomas B. Moeslund
    links: []
- ref_id: ref-7
  citation: '[7] A. Cioppa, A. Deliège, and M. Van Droogenbroeck. A bottom-up approach
    based on semantics for the interpretation of the main camera stream in soccer
    games. In IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops
    (CVPRW), pages 1846–184609, 2018. ISSN: 2160-7516.'
  target_id: cite-dcff37407f36f742
  identification_status: resolved
  roles:
  - related
  decision_status: classified
  reason: 'スポーツ映像における行動の局在化・分類（activity localization and classification）の先行研究として§2で言及されている。

    Identification: OpenAlex referenced_works: unique_title_year_first_author_match'
  evidence:
  - Section 2, anchor S2.p1.1
  - bib [7] (anchor bib.bib7)
  lookup:
    title: A bottom-up approach based on semantics for the interpretation of the main
      camera stream in soccer games
    year: 2018
    authors:
    - A. Cioppa
    - A. Deliège
    - M. Van Droogenbroeck
    links: []
- ref_id: ref-8
  citation: '[8] Caroline Cohen, Baptiste Darbois Texier, David Quéré, and Christophe
    Clanet. The physics of badminton. New Journal of Physics, 17(6):063001, June 2015.'
  target_id: cite-90b70d5abcde69ae
  identification_status: resolved
  roles:
  - component
  decision_status: classified
  reason: '3D軌道再構成（§4.5）において、シャトルの飛行を空気抵抗を受ける質点運動の微分方程式（式(1)・式(2)）で近似するための物理モデルとして採用されている。

    Identification: OpenAlex referenced_works: unique_title_year_first_author_match'
  evidence:
  - Section 4.5 (Physics-based trajectory estimation), Equation 1-2, anchor S4.SS5.SSS0.Px1.p1.1
  - bib [8] (anchor bib.bib8)
  lookup:
    title: The physics of badminton
    year: 2015
    authors:
    - Caroline Cohen
    - Baptiste Darbois Texier
    - David Quéré
    - Christophe Clanet
    links: []
- ref_id: ref-9
  citation: '[9] MMPose Contributors. Openmmlab pose estimation toolbox and benchmark.
    https://github.com/open-mmlab/mmpose, 2020.'
  target_id: cite-5b2467acdee75b41
  identification_status: resolved
  roles:
  - component
  decision_status: classified
  reason: 'システムの選手姿勢推定ステップ（§4.2）において、HRNetモデルで各フレームの選手姿勢を算出するためのツールボックス／フレームワーク（mmpose）として採用されている。

    Identification: explicit external ID in reference'
  evidence:
  - Section 4.2, anchor S4.SS2.p1.1
  - bib [9] (anchor bib.bib9)
  lookup:
    title: Openmmlab pose estimation toolbox and benchmark
    year: 2020
    authors:
    - MMPose Contributors
    links:
    - https://github.com/open-mmlab/mmpose
- ref_id: ref-10
  citation: '[10] ESPN. Badminton second to soccer in participation worldwide, 2004.'
  target_id: null
  identification_status: unresolved
  roles: []
  decision_status: unclassifiable
  reason: '§1冒頭でバドミントンが世界で2番目に競技人口の多いスポーツであるという一般的背景の根拠としてのみ言及されており、同タスクの先行研究や技術的採用・比較・ベンチマークとしての役割を示す記述がないため分類不能。

    Identification: OpenAlex referenced_works (33 candidates): no_title_match'
  evidence:
  - Section 1, anchor S1.p1.1
  - bib [10] (anchor bib.bib10)
  lookup:
    title: Badminton second to soccer in participation worldwide
    year: 2004
    authors:
    - ESPN
    links: []
- ref_id: ref-11
  citation: '[11] Dirk Farin, Susanne Krabbe, Peter H. N. de With, and Wolfgang Effelsberg.
    Robust camera calibration for sport videos using court models. In Minerva M. Yeung,
    Rainer Lienhart, and Chung-Sheng Li, editors, Storage and Retrieval Methods and
    Applications for Multimedia 2004, San Jose, CA, USA, January 20, 2004, volume
    5307 of SPIE Proceedings, pages 80–91. SPIE, 2004.'
  target_id: null
  identification_status: unresolved
  roles:
  - foundation
  - comparison
  - related
  decision_status: classified
  reason: '§2でコート検出の先行研究として挙げられ、§4.1では提案コート検出手法の出発点となるアルゴリズム（foundation）として採用して線分分割をグラフベース手法に改良し、Figure
    3および§4.1の定量評価（検出成功率73.9%→85.5%、平均IoU 0.96→0.97、処理時間1/40）で直接比較している。

    Identification: OpenAlex referenced_works (33 candidates): author_mismatch_or_missing'
  evidence:
  - Section 2, anchor S2.p1.1
  - Section 3, Figure 3 (anchor S3.F3.8)
  - Section 4.1, anchor S4.SS1.p1.1, S4.SS1.p2.1, S4.SS1.p4.1
  - bib [11] (anchor bib.bib11)
  lookup:
    title: Robust camera calibration for sport videos using court models
    year: 2004
    authors:
    - Dirk Farin
    - Susanne Krabbe
    - Peter H. N. de With
    - Wolfgang Effelsberg
    links: []
- ref_id: ref-12
  citation: '[12] Rohit Girdhar, Georgia Gkioxari, Lorenzo Torresani, Manohar Paluri,
    and Du Tran. Detect-and-Track: Efficient Pose Estimation in Videos. arXiv:1712.09184
    [cs], May 2018. arXiv: 1712.09184.'
  target_id: cite-ddea9a4362709b55
  identification_status: resolved
  roles:
  - related
  decision_status: classified
  reason: '§4.2の姿勢追跡において、コート境界を利用する提案手法とは別の非構造化環境向け姿勢追跡の先行研究として言及されているのみで、採用や定量比較は行われていない。

    Identification: explicit external ID in reference'
  evidence:
  - Section 4.2, anchor S4.SS2.p1.1
  - bib [12] (anchor bib.bib12)
- ref_id: ref-13
  citation: '[13] Geoffrey E. Hinton, Oriol Vinyals, and Jeffrey Dean. Distilling
    the knowledge in a neural network. CoRR, abs/1503.02531, 2015.'
  target_id: cite-61436ddfdddf30b0
  identification_status: resolved
  roles:
  - component
  decision_status: classified
  reason: 'シャトル検出モデル（§4.3）の学習初期（最初の10エポック）において、TrackNetV2のパラメータを用いてモデルを学習させる知識蒸留（distillation
    learning）技法として採用されている。

    Identification: OpenAlex referenced_works: unique_title_year_first_author_match'
  evidence:
  - Section 4.3, anchor S4.SS3.p2.1
  - bib [13] (anchor bib.bib13)
  lookup:
    title: Distilling the knowledge in a neural network
    year: 2015
    authors:
    - Geoffrey E. Hinton
    - Oriol Vinyals
    - Jeffrey Dean
    links: []
- ref_id: ref-14
  citation: '[14] Namdar Homayounfar, Sanja Fidler, and Raquel Urtasun. Sports field
    localization via deep structured models. In IEEE Conference on Computer Vision
    and Pattern Recognition (CVPR), pages 4012–4020. IEEE, 2017.'
  target_id: cite-9266087073b249e5
  identification_status: resolved
  roles:
  - related
  decision_status: classified
  reason: 'スポーツ映像におけるコート／フィールド検出の先行研究として§2で言及されている。

    Identification: OpenAlex referenced_works: unique_title_year_first_author_match'
  evidence:
  - Section 2, anchor S2.p1.1
  - bib [14] (anchor bib.bib14)
  lookup:
    title: Sports field localization via deep structured models
    year: 2017
    authors:
    - Namdar Homayounfar
    - Sanja Fidler
    - Raquel Urtasun
    links: []
- ref_id: ref-15
  citation: '[15] James Hong, Matthew Fisher, Michaël Gharbi, and Kayvon Fatahalian.
    Video pose distillation for few-shot, fine-grained sports action recognition.
    CoRR, abs/2109.01305, 2021.'
  target_id: cite-8d5a358fae30ba99
  identification_status: resolved
  roles:
  - related
  decision_status: classified
  reason: 'スポーツ映像における姿勢解析（pose analysis）の先行研究として§2で言及されている。

    Identification: OpenAlex referenced_works: unique_title_year_first_author_match'
  evidence:
  - Section 2, anchor S2.p1.1
  - bib [15] (anchor bib.bib15)
  lookup:
    title: Video pose distillation for few-shot, fine-grained sports action recognition
    year: 2021
    authors:
    - James Hong
    - Matthew Fisher
    - Michaël Gharbi
    - Kayvon Fatahalian
    links: []
- ref_id: ref-16
  citation: '[16] Yu-Chuan Huang, I.-No Liao, Ching-Hsuan Chen, Tsì-Uí İk, and Wen-Chih
    Peng. TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects
    in Sports Applications. arXiv:1907.03698 [cs, stat], July 2019. arXiv: 1907.03698.'
  target_id: cite-b33241131d4249b9
  identification_status: resolved
  roles:
  - related
  - benchmark
  decision_status: classified
  reason: '§2でボール/シャトル追跡の先行研究として言及され、§3ではTrackNetV2データセットとともに引用され、3つのテスト試合（12kフレーム）を評価に用いるTrackNetの分割プロトコルに従うデータセット基盤として利用されている。

    Identification: explicit external ID in reference'
  evidence:
  - Section 2, anchor S2.p1.1
  - Section 3, anchor S3.p1.1
  - bib [16] (anchor bib.bib16)
- ref_id: ref-17
  citation: '[17] Hyunwoo Kim and Ki Sang Hong. Soccer video mosaicing using self-calibration
    and line tracking. In Proceedings 15th International Conference on Pattern Recognition.
    ICPR, volume 1, pages 592–595 vol.1, 2000. ISSN: 1051-4651.'
  target_id: null
  identification_status: unresolved
  roles:
  - related
  decision_status: classified
  reason: 'スポーツ映像におけるコート／フィールド検出の先行研究として§2で言及されている。

    Identification: OpenAlex referenced_works (33 candidates): year_mismatch_or_missing'
  evidence:
  - Section 2, anchor S2.p1.1
  - bib [17] (anchor bib.bib17)
  lookup:
    title: Soccer video mosaicing using self-calibration and line tracking
    year: 2000
    authors:
    - Hyunwoo Kim
    - Ki Sang Hong
    links: []
- ref_id: ref-18
  citation: '[18] Kaustubh Milind Kulkarni and Sucheth Shenoy. Table Tennis Stroke
    Recognition Using Two-Dimensional Human Pose Estimation. In 2021 IEEE/CVF Conference
    on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 4571–4579,
    Nashville, TN, USA, June 2021. IEEE.'
  target_id: cite-e9044189be07413f
  identification_status: resolved
  roles:
  - related
  decision_status: classified
  reason: 'ラケットスポーツにおけるストローク解析（stroke analysis）の先行研究として§2で言及されている。

    Identification: OpenAlex referenced_works: unique_title_year_first_author_match'
  evidence:
  - Section 2, anchor S2.p1.1
  - bib [18] (anchor bib.bib18)
  lookup:
    title: Table Tennis Stroke Recognition Using Two-Dimensional Human Pose Estimation
    year: 2021
    authors:
    - Kaustubh Milind Kulkarni
    - Sucheth Shenoy
    links: []
- ref_id: ref-19
  citation: '[19] Chia-Lo Lee and Peter J Ramadge. Badminton Shuttlecock Tracking
    and 3D Trajectory Estimation From Video. Princeton DataSpace, Electrical Engineering,
    page 69, 2019.'
  target_id: null
  identification_status: unresolved
  roles:
  - related
  decision_status: classified
  reason: 'バドミントン映像からのシャトル追跡およびモデルベース3D軌道推定を行う先行研究として§1・§2で言及されている（2D合成軌道のみで評価されている旨が述べられているが、定量的な実験比較は実施されていない）。

    Identification: OpenAlex referenced_works (33 candidates): year_mismatch_or_missing'
  evidence:
  - Section 1, anchor S1.p3.1
  - Section 2, anchor S2.p2.1
  - bib [19] (anchor bib.bib19)
  lookup:
    title: Badminton Shuttlecock Tracking and 3D Trajectory Estimation From Video
    year: 2019
    authors:
    - Chia-Lo Lee
    - Peter J Ramadge
    links: []
- ref_id: ref-20
  citation: '[20] Yang Liu, Dawei Liang, Qingming Huang, and Wen Gao. Extracting 3d
    information from broadcast soccer video. Image Vis. Comput., 24(10):1146–1162,
    2006.'
  target_id: cite-8ed68885a64fecfe
  identification_status: resolved
  roles:
  - related
  decision_status: classified
  reason: 'コート検出（§4.1）において、Farinらのモデルベースコート検出アルゴリズムに関連する放送スポーツ映像の先行研究群の1つとして引用されている。

    Identification: OpenAlex referenced_works: unique_title_year_first_author_match'
  evidence:
  - Section 4.1, anchor S4.SS1.p1.1
  - bib [20] (anchor bib.bib20)
  lookup:
    title: Extracting 3d information from broadcast soccer video
    year: 2006
    authors:
    - Yang Liu
    - Dawei Liang
    - Qingming Huang
    - Wen Gao
    links: []
- ref_id: ref-21
  citation: '[21] Amir Sadeghian, Alexandre Alahi, and Silvio Savarese. Tracking the
    untrackable: Learning to track multiple cues with long-term dependencies. In IEEE
    International Conference on Computer Vision, ICCV 2017, Venice, Italy, October
    22-29, 2017, pages 300–311. IEEE Computer Society, 2017.'
  target_id: cite-5f3bd0e0bf1ac9b8
  identification_status: resolved
  roles:
  - related
  decision_status: classified
  reason: '§4.2の姿勢追跡において、コート手がかりを用いる提案手法が採用しないリカレントネットワークベースの既存追跡手法として言及されているのみで、採用や実験比較は行われていない。

    Identification: OpenAlex referenced_works: unique_title_year_first_author_match'
  evidence:
  - Section 4.2, anchor S4.SS2.p1.1
  - bib [21] (anchor bib.bib21)
  lookup:
    title: 'Tracking the untrackable: Learning to track multiple cues with long-term
      dependencies'
    year: 2017
    authors:
    - Amir Sadeghian
    - Alexandre Alahi
    - Silvio Savarese
    links: []
- ref_id: ref-22
  citation: '[22] Saikat Sarkar, Amlan Chakrabarti, and Dipti Prasad Mukherjee. Generation
    of ball possession statistics in soccer using minimum-cost flow network. In IEEE/CVF
    Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages
    2515–2523. IEEE, 2019.'
  target_id: cite-c191855a0bf3b73a
  identification_status: resolved
  roles:
  - related
  decision_status: classified
  reason: 'スポーツ映像におけるボール追跡（ball tracking）の先行研究として§2で言及されている。

    Identification: OpenAlex referenced_works: unique_title_year_first_author_match'
  evidence:
  - Section 2, anchor S2.p1.1
  - bib [22] (anchor bib.bib22)
  lookup:
    title: Generation of ball possession statistics in soccer using minimum-cost flow
      network
    year: 2019
    authors:
    - Saikat Sarkar
    - Amlan Chakrabarti
    - Dipti Prasad Mukherjee
    links: []
- ref_id: ref-23
  citation: '[23] Steven Schwarcz, Peng Xu, David D’Ambrosio, Juhana Kangaspunta,
    Anelia Angelova, Huong Phan, and Navdeep Jaitly. SPIN: A High Speed, High Resolution
    Vision Dataset for Tracking and Action Recognition in Ping Pong. arXiv:1912.06640
    [cs], Dec. 2019. arXiv: 1912.06640.'
  target_id: cite-6445fa32fd43134a
  identification_status: resolved
  roles:
  - related
  decision_status: classified
  reason: 'ラケットスポーツ映像における行動の局在化・分類（activity localization and classification）の先行研究として§2で言及されている。

    Identification: explicit external ID in reference'
  evidence:
  - Section 2, anchor S2.p1.1
  - bib [23] (anchor bib.bib23)
- ref_id: ref-24
  citation: '[24] Lejun Shen, Qing Liu, Lin Li, and Yawei Ren. Reconstruction of 3D
    Ball/Shuttle Position by Two Image Points from a Single View. In Martin Lames,
    Dietmar Saupe, and Josef Wiemeyer, editors, Proceedings of the 11th International
    Symposium on Computer Science in Sport (IACSS 2017), volume 663, pages 89–96.
    Springer International Publishing, Cham, 2018. Series Title: Advances in Intelligent
    Systems and Computing.'
  target_id: null
  identification_status: unresolved
  roles:
  - related
  decision_status: classified
  reason: '単一視点映像からconfirming pointsを用いて3Dシャトル位置を再構成する先行研究（全ショットで人手介入を必要とする手法）として§1および§2で言及されている。

    Identification: OpenAlex referenced_works (33 candidates): year_mismatch_or_missing'
  evidence:
  - Section 1, anchor S1.p3.1
  - Section 2, anchor S2.p2.1
  - bib [24] (anchor bib.bib24)
  lookup:
    title: Reconstruction of 3D Ball/Shuttle Position by Two Image Points from a Single
      View
    year: 2018
    authors:
    - Lejun Shen
    - Qing Liu
    - Lin Li
    - Yawei Ren
    links: []
- ref_id: ref-25
  citation: '[25] Lejun Shen, Hui Zhang, Min Zhu, Jia Zheng, and Yawei Ren. Measurement
    and Performance Evaluation of Lob Technique Using Aerodynamic Model in Badminton
    Matches. In Martin Lames, Alexander Danilov, Egor Timme, and Yuri Vassilevski,
    editors, Proceedings of the 12th International Symposium on Computer Science in
    Sport (IACSS 2019), volume 1028, pages 53–58. Springer International Publishing,
    Cham, 2020. Series Title: Advances in Intelligent Systems and Computing.'
  target_id: null
  identification_status: unresolved
  roles:
  - related
  decision_status: classified
  reason: 'バドミントン試合において空気力学モデルとconfirming pointsで3Dシャトル軌道を推定する先行研究（ショット単位の人手介入が必要）として§1および§2で言及されている。

    Identification: OpenAlex referenced_works (33 candidates): year_mismatch_or_missing'
  evidence:
  - Section 1, anchor S1.p3.1
  - Section 2, anchor S2.p2.1
  - bib [25] (anchor bib.bib25)
  lookup:
    title: Measurement and Performance Evaluation of Lob Technique Using Aerodynamic
      Model in Badminton Matches
    year: 2020
    authors:
    - Lejun Shen
    - Hui Zhang
    - Min Zhu
    - Jia Zheng
    - Yawei Ren
    links: []
- ref_id: ref-26
  citation: '[26] N.-E. Sun, Y.-C. Lin, S.-P. Chuang, T.-H. Hsu, D.-R. Yu, H.-Y. Chung,
    and T.-U. İk. TrackNetV2: Efficient Shuttlecock Tracking Network. In 2020 International
    Conference on Pervasive Artificial Intelligence (ICPAI), pages 86–91, Dec. 2020.'
  target_id: cite-bee887c491b4ab99
  identification_status: resolved
  roles:
  - foundation
  - comparison
  - related
  - benchmark
  decision_status: classified
  reason: '§2でシャトル追跡の先行研究として挙げられ、§3ではTrackNetV2データセットを学習・評価データ（benchmark）として採用している。さらに§4.3では3-in-3-outのU-Net構成を出発点（foundation）とし、そのパラメータを初期エポックの知識蒸留にも用いる一方、パラメータ数（2.9M対11.3M）および精度（88.6%対84.0%）を直接比較（comparison）している。

    Identification: OpenAlex referenced_works: unique_title_year_first_author_match'
  evidence:
  - Section 2, anchor S2.p1.1
  - Section 3, anchor S3.p1.1
  - Section 4.3, anchor S4.SS3.p1.1, S4.SS3.p2.1
  - bib [26] (anchor bib.bib26)
  lookup:
    title: 'TrackNetV2: Efficient Shuttlecock Tracking Network'
    year: 2020
    authors:
    - N.-E. Sun
    - Y.-C. Lin
    - S.-P. Chuang
    - T.-H. Hsu
    - D.-R. Yu
    - H.-Y. Chung
    - T.-U. İk
    links: []
- ref_id: ref-27
  citation: '[27] Richard Szeliski. Computer Vision: Algorithms and Applications.
    Springer-Verlag, 1st edition, 2010.'
  target_id: null
  identification_status: unresolved
  roles: []
  decision_status: unclassifiable
  reason: '§2で「3D再構成はコンピュータビジョン分野で広く研究されている」という一般的背景説明のために標準的教科書として引用されているのみで、同タスクの先行研究や具体的技術の採用・比較・ベンチマークとしての役割を示す記述がないため分類不能。

    Identification: OpenAlex referenced_works (33 candidates): year_mismatch_or_missing'
  evidence:
  - Section 2, anchor S2.p2.1
  - bib [27] (anchor bib.bib27)
  lookup:
    title: 'Computer Vision: Algorithms and Applications'
    year: 2010
    authors:
    - Richard Szeliski
    links: []
- ref_id: ref-28
  citation: '[28] Saeid Asgari Taghanaki, Yefeng Zheng, Shaohua Kevin Zhou, Bogdan
    Georgescu, Puneet Sharma, Daguang Xu, Dorin Comaniciu, and Ghassan Hamarneh. Combo
    loss: Handling input and output imbalance in multi-organ segmentation. Comput.
    Medical Imaging Graph., 75:24–33, 2019.'
  target_id: cite-173ea2a09a4a1c31
  identification_status: resolved
  roles:
  - component
  decision_status: classified
  reason: 'シャトル検出モデル（§4.3）において、微小なシャトルによる入力不均衡問題に対処するため、Dice損失とバイナリ交差エントロピー損失の重み付き組み合わせ（Combo
    loss）を損失関数として採用している。

    Identification: OpenAlex referenced_works: unique_title_year_first_author_match'
  evidence:
  - Section 4.3, anchor S4.SS3.p1.1
  - bib [28] (anchor bib.bib28)
  lookup:
    title: 'Combo loss: Handling input and output imbalance in multi-organ segmentation'
    year: 2019
    authors:
    - Saeid Asgari Taghanaki
    - Yefeng Zheng
    - Shaohua Kevin Zhou
    - Bogdan Georgescu
    - Puneet Sharma
    - Daguang Xu
    - Dorin Comaniciu
    - Ghassan Hamarneh
    links: []
- ref_id: ref-29
  citation: '[29] Rajkumar Theagarajan, Federico Pala, Xiu Zhang, and Bir Bhanu. Soccer:
    Who has the ball? generating visual analytics and player statistics. In IEEE/CVF
    Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages
    1830–18308, 2018. ISSN: 2160-7516.'
  target_id: cite-04fd0207abe489ab
  identification_status: resolved
  roles:
  - related
  decision_status: classified
  reason: 'スポーツ映像におけるボール追跡（ball tracking）の先行研究として§2で言及されている。

    Identification: OpenAlex referenced_works: unique_title_year_first_author_match'
  evidence:
  - Section 2, anchor S2.p1.1
  - bib [29] (anchor bib.bib29)
  lookup:
    title: 'Soccer: Who has the ball? generating visual analytics and player statistics'
    year: 2018
    authors:
    - Rajkumar Theagarajan
    - Federico Pala
    - Xiu Zhang
    - Bir Bhanu
    links: []
- ref_id: ref-30
  citation: '[30] Takamasa Tsunoda, Yasuhiro Komori, Masakazu Matsugu, and Tatsuya
    Harada. Football action recognition using hierarchical LSTM. In IEEE Conference
    on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 155–163. IEEE,
    2017.'
  target_id: cite-3e4e9b5bacb6beee
  identification_status: resolved
  roles:
  - related
  decision_status: classified
  reason: 'スポーツ映像における行動の局在化・分類（activity localization and classification）の先行研究として§2で言及されている。

    Identification: OpenAlex referenced_works: unique_title_year_first_author_match'
  evidence:
  - Section 2, anchor S2.p1.1
  - bib [30] (anchor bib.bib30)
  lookup:
    title: Football action recognition using hierarchical LSTM
    year: 2017
    authors:
    - Takamasa Tsunoda
    - Yasuhiro Komori
    - Masakazu Matsugu
    - Tatsuya Harada
    links: []
- ref_id: ref-31
  citation: '[31] Fei Wang, Lifeng Sun, Bo Yang, and Shiqiang Yang. Fast arc detection
    algorithm for play field registration in soccer video mining. In 2006 IEEE International
    Conference on Systems, Man and Cybernetics, volume 6, pages 4932–4936, 2006. ISSN:
    1062-922X.'
  target_id: cite-178d98985ca8aa47
  identification_status: resolved
  roles:
  - related
  decision_status: classified
  reason: 'スポーツ映像におけるコート／フィールド検出の先行研究として§2で言及されている。

    Identification: OpenAlex referenced_works: unique_title_year_first_author_match'
  evidence:
  - Section 2, anchor S2.p1.1
  - bib [31] (anchor bib.bib31)
  lookup:
    title: Fast arc detection algorithm for play field registration in soccer video
      mining
    year: 2006
    authors:
    - Fei Wang
    - Lifeng Sun
    - Bo Yang
    - Shiqiang Yang
    links: []
- ref_id: ref-32
  citation: '[32] Jingdong Wang, Ke Sun, Tianheng Cheng, Borui Jiang, Chaorui Deng,
    Yang Zhao, Dong Liu, Yadong Mu, Mingkui Tan, Xinggang Wang, Wenyu Liu, and Bin
    Xiao. Deep high-resolution representation learning for visual recognition. IEEE
    Trans. Pattern Anal. Mach. Intell., 43(10):3349–3364, 2021.'
  target_id: null
  identification_status: unresolved
  roles:
  - component
  decision_status: classified
  reason: 'システムの選手姿勢推定ステップ（§4.2）において、各フレームの選手姿勢を推定するためのトップダウン型モデル（HRNet）として採用されている。

    Identification: OpenAlex referenced_works (33 candidates): year_mismatch_or_missing'
  evidence:
  - Section 4.2, anchor S4.SS2.p1.1
  - bib [32] (anchor bib.bib32)
  lookup:
    title: Deep high-resolution representation learning for visual recognition
    year: 2021
    authors:
    - Jingdong Wang
    - Ke Sun
    - Tianheng Cheng
    - Borui Jiang
    - Chaorui Deng
    - Yang Zhao
    - Dong Liu
    - Yadong Mu
    - Mingkui Tan
    - Xinggang Wang
    - Wenyu Liu
    - Bin Xiao
    links: []
- ref_id: ref-33
  citation: '[33] T. Watanabe, M. Haseyama, and H. Kitajima. A soccer field tracking
    method with wire frame model from TV images. In International Conference on Image
    Processing, 2004. ICIP ’04., volume 3, pages 1633–1636 Vol. 3, 2004. ISSN: 1522-4880.'
  target_id: null
  identification_status: unresolved
  roles:
  - related
  decision_status: classified
  reason: 'スポーツ映像におけるコート／フィールド検出の先行研究として§2で言及されている。

    Identification: OpenAlex referenced_works (33 candidates): year_mismatch_or_missing'
  evidence:
  - Section 2, anchor S2.p1.1
  - bib [33] (anchor bib.bib33)
  lookup:
    title: A soccer field tracking method with wire frame model from TV images
    year: 2004
    authors:
    - T. Watanabe
    - M. Haseyama
    - H. Kitajima
    links: []
- ref_id: ref-34
  citation: '[34] A. Yamada, Y. Shirai, and J. Miura. Tracking players and a ball
    in video image sequence and estimating camera parameters for 3d interpretation
    of soccer games. In International Conference on Pattern Recognition, volume 1,
    pages 303–306 vol.1, 2002. ISSN: 1051-4651.'
  target_id: null
  identification_status: unresolved
  roles:
  - related
  decision_status: classified
  reason: 'スポーツ映像におけるコート検出およびカメラパラメータ推定の先行研究として§2で言及されている。

    Identification: OpenAlex referenced_works (33 candidates): year_mismatch_or_missing'
  evidence:
  - Section 2, anchor S2.p1.1
  - bib [34] (anchor bib.bib34)
  lookup:
    title: Tracking players and a ball in video image sequence and estimating camera
      parameters for 3d interpretation of soccer games
    year: 2002
    authors:
    - A. Yamada
    - Y. Shirai
    - J. Miura
    links: []
- ref_id: ref-35
  citation: '[35] Shuainan Ye, Zhutian Chen, Xiangtong Chu, Yifan Wang, Siwei Fu,
    Lejun Shen, Kun Zhou, and Yingcai Wu. ShuttleSpace: Exploring and Analyzing Movement
    Trajectory in Immersive Visualization. IEEE Transactions on Visualization and
    Computer Graphics, 27(2):860–869, Feb. 2021.'
  target_id: null
  identification_status: unresolved
  roles:
  - related
  decision_status: classified
  reason: 'バドミントンの3D軌道を没入型可視化により戦術分析へ活用する先行研究（confirming pointsによる人手介入が必要）として§1および§2で言及されている。

    Identification: OpenAlex referenced_works (33 candidates): year_mismatch_or_missing'
  evidence:
  - Section 1, anchor S1.p2.1
  - Section 2, anchor S2.p1.1, S2.p2.1
  - bib [35] (anchor bib.bib35)
  lookup:
    title: 'ShuttleSpace: Exploring and Analyzing Movement Trajectory in Immersive
      Visualization'
    year: 2021
    authors:
    - Shuainan Ye
    - Zhutian Chen
    - Xiangtong Chu
    - Yifan Wang
    - Siwei Fu
    - Lejun Shen
    - Kun Zhou
    - Yingcai Wu
    links: []
- ref_id: ref-36
  citation: '[36] Haotian Zhang, Cristobal Sciutto, Maneesh Agrawala, and Kayvon Fatahalian.
    Vid2Player: Controllable Video Sprites that Behave and Appear like Professional
    Tennis Players. arXiv:2008.04524 [cs], Dec. 2020. arXiv: 2008.04524.'
  target_id: cite-e2061bb273c6828c
  identification_status: resolved
  roles:
  - related
  decision_status: classified
  reason: 'テニス映像における弾道3D軌道再構成およびラリーの循環的ショット構造を扱う先行研究（ショット境界は手動アノテーションを利用）として§1および§2で言及されている。

    Identification: explicit external ID in reference'
  evidence:
  - Section 1, anchor S1.p3.1
  - Section 2, anchor S2.p1.1, S2.p2.1
  - bib [36] (anchor bib.bib36)
schema_version: 1
---
Reader: agy 1.5.9, gemini-4-hydrogen-high (observed: Gemini 4 Hydrogen (High)). Reading elapsed: 127.620 seconds. Input: html, https://arxiv.org/html/2204.01899.

Legacy identification results invalidated; original reading and classifications retained.
