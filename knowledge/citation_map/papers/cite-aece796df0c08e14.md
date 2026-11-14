---
id: cite-aece796df0c08e14
type: citation_paper
title: 'DeepBall: Deep Neural-Network Ball Detector'
identity_key: doi:10.5220/0007348900002108
year: 2019
authors:
- Jacek Komorowski
- Grzegorz Kurzejamski
- Grzegorz Sarwas
external_ids:
  openalex: W2915174681
  doi: 10.5220/0007348900002108
  arxiv: '1902.07304'
source: https://arxiv.org/html/1902.07304
summary: 本論文は、サッカーの長距離撮影映像からボールを高速かつ高精度に検出する全畳み込みニューラルネットワーク「DeepBall」を提案している。複数階層の畳み込み特徴マップをアップサンプリングして結合するハイパーカラム構造を採用することで、視覚的コンテキストを活用しつつ任意解像度の入力画像からボール位置を示す信頼度マップを1パスで生成する。ISSIA-CNR
  Soccer Datasetを用いた評価実験では、既存のニューラルネットワーク手法（SpeckらおよびRenoら）を上回るAP 0.877・精度0.951を達成し、パラメータ数を抑えつつ190
  FPSでのリアルタイム処理を実現した。
text_status: available
text_reason: 'Identity verified: arXiv ID 1902.07304 and exact HTML title match stored
  reading (review audit; no agy rereading).'
references:
- ref_id: ref-1
  citation: Ali, M. N., Abdullah-Al-Wadud, M., and Lee, S.-L. (2012). An efficient
    algorithm for detection of soccer ball and players. Proc. 16th ASTL Control and
    Networking, 16.
  target_id: cite-a8a891a6805429d3
  identification_status: resolved
  roles:
  - related
  decision_status: classified
  reason: '関連研究（Section 2）において、色特徴に基づく背景差分およびカーネルベースHough変換によるピッチ白線除去を用いた従来のサッカーボール検出手法として言及されている。

    Identification: OpenAlex referenced_works: unique_title_year_first_author_match'
  evidence:
  - Section 2 (S2.p1.1, S2.p3.1)
  - Bibliography (bib.bibx1)
  lookup:
    title: An efficient algorithm for detection of soccer ball and players
    year: 2012
    authors:
    - M. N. Ali
    - M. Abdullah-Al-Wadud
    - S.-L. Lee
    links: []
- ref_id: ref-2
  citation: D’Orazio, T., Ancona, N., Cicirelli, G., and Nitti, M. (2002). A ball
    detection algorithm for real soccer image sequences. In Pattern Recognition, 2002.
    Proceedings. 16th International Conference on, volume 1, pages 210–213. IEEE.
  target_id: null
  identification_status: unresolved
  roles:
  - related
  decision_status: classified
  reason: '関連研究（Section 2）において、動き検出に基づく背景差分、球状物体向けの変形円Hough変換、および2段階候補抽出・検証を用いる従来のボール検出手法として言及されている。

    Identification: OpenAlex referenced_works (24 candidates): year_mismatch_or_missing'
  evidence:
  - Section 2 (S2.p1.1, S2.p2.1)
  - Bibliography (bib.bibx2)
  lookup:
    title: A ball detection algorithm for real soccer image sequences
    year: 2002
    authors:
    - T. D’Orazio
    - N. Ancona
    - G. Cicirelli
    - M. Nitti
    links: []
- ref_id: ref-3
  citation: D’Orazio, T., Guaragnella, C., Leo, M., and Distante, A. (2004). A new
    algorithm for ball recognition using circle hough transform and neural classifier.
    Pattern Recognition, 37(3):393 – 408.
  target_id: null
  identification_status: unresolved
  roles:
  - related
  decision_status: classified
  reason: '関連研究（Section 2）において、動き検出に基づく背景差分と変形円Hough変換を用いる従来のサッカーボール検出手法として言及されている。

    Identification: OpenAlex referenced_works (24 candidates): year_mismatch_or_missing'
  evidence:
  - Section 2 (S2.p1.1, S2.p2.1)
  - Bibliography (bib.bibx3)
  lookup:
    title: A new algorithm for ball recognition using circle hough transform and neural
      classifier
    year: 2004
    authors:
    - T. D’Orazio
    - C. Guaragnella
    - M. Leo
    - A. Distante
    links: []
- ref_id: ref-4
  citation: D’Orazio, T., Leo, M., Mosca, N., Spagnolo, P., and Mazzeo, P. (2009).
    A semi-automatic system for ground truth generation of soccer video sequences.
    In 2009 Advanced Video and Signal Based Surveillance, pages 559–564. IEEE.
  target_id: cite-fc7317d358322ac4
  identification_status: resolved
  roles:
  - benchmark
  decision_status: classified
  reason: '本論文が学習および評価に使用した公開データセット（ISSIA-CNR Soccer Dataset）の出典文献である。

    Identification: OpenAlex referenced_works: unique_title_year_first_author_match'
  evidence:
  - Section 3 Training dataset (S3.SS0.SSS0.Px3.p1.1)
  - Section 4 Evaluation dataset (S4.SS0.SSS0.Px1.p1.1)
  - Section 5 (S5.p1.1)
  - Bibliography (bib.bibx4)
  lookup:
    title: A semi-automatic system for ground truth generation of soccer video sequences
    year: 2009
    authors:
    - T. D’Orazio
    - M. Leo
    - N. Mosca
    - P. Spagnolo
    - P. Mazzeo
    links: []
- ref_id: ref-5
  citation: Everingham, M., Van Gool, L., Williams, C. K., Winn, J., and Zisserman,
    A. (2010). The pascal visual object classes (voc) challenge. International journal
    of computer vision, 88(2):303–338.
  target_id: null
  identification_status: unresolved
  roles:
  - benchmark
  decision_status: classified
  reason: '評価指標であるAverage Precision（AP）の定義として、Pascal 2007 VOC Challengeの評価プロトコルを採用している。

    Identification: OpenAlex referenced_works (24 candidates): year_mismatch_or_missing'
  evidence:
  - Section 4 Evaluation metrics (S4.SS0.SSS0.Px2.p1.1)
  - Bibliography (bib.bibx5)
  lookup:
    title: The pascal visual object classes (voc) challenge
    year: 2010
    authors:
    - M. Everingham
    - L. Van Gool
    - C. K. Williams
    - J. Winn
    - A. Zisserman
    links: []
- ref_id: ref-6
  citation: Felzenszwalb, P. F., Girshick, R. B., McAllester, D., and Ramanan, D.
    (2010). Object detection with discriminatively trained part-based models. IEEE
    transactions on pattern analysis and machine intelligence, 32(9):1627–1645.
  target_id: null
  identification_status: unresolved
  roles:
  - related
  decision_status: classified
  reason: '関連研究（Section 2）において、深層学習ベースの物体検出器と対比される従来型の物体検出手法（DPM）として言及されている。

    Identification: OpenAlex referenced_works (24 candidates): year_mismatch_or_missing'
  evidence:
  - Section 2 (S2.p5.1)
  - Bibliography (bib.bibx6)
  lookup:
    title: Object detection with discriminatively trained part-based models
    year: 2010
    authors:
    - P. F. Felzenszwalb
    - R. B. Girshick
    - D. McAllester
    - D. Ramanan
    links: []
- ref_id: ref-7
  citation: Girshick, R. (2015). Fast r-cnn. In Proceedings of the IEEE international
    conference on computer vision, pages 1440–1448.
  target_id: null
  identification_status: unresolved
  roles:
  - related
  decision_status: classified
  reason: '関連研究（Section 2）において、2段階の物体検出器（Fast R-CNN）の代表例として言及されている。

    Identification: OpenAlex referenced_works (24 candidates): no_title_match'
  evidence:
  - Section 2 (S2.p5.1)
  - Bibliography (bib.bibx7)
  lookup:
    title: Fast r-cnn
    year: 2015
    authors:
    - R. Girshick
    links: []
- ref_id: ref-8
  citation: Gong, Y., Sin, L. T., Chuan, C. H., Zhang, H., and Sakauchi, M. (1995).
    Automatic parsing of tv soccer programs. In Proceedings of the International Conference
    on Multimedia Computing and Systems, pages 167–174.
  target_id: null
  identification_status: unresolved
  roles:
  - related
  decision_status: classified
  reason: '関連研究（Section 2）において、色特徴による背景差分、ブロブの形状・サイズ等のヒューリスティック基準、および連続フレームを用いた検証を行う従来のサッカーボール検出手法として言及されている。

    Identification: OpenAlex referenced_works (24 candidates): year_mismatch_or_missing'
  evidence:
  - Section 2 (S2.p1.1, S2.p2.1, S2.p3.1)
  - Bibliography (bib.bibx8)
  lookup:
    title: Automatic parsing of tv soccer programs
    year: 1995
    authors:
    - Y. Gong
    - L. T. Sin
    - C. H. Chuan
    - H. Zhang
    - M. Sakauchi
    links: []
- ref_id: ref-9
  citation: Halbinger, J. and Metzler, J. (2015). Video-based soccer ball detection
    in difficult situations. In Cabri, J., Pezarat Correia, P., and Barreiros, J.,
    editors, Sports Science Research and Technology Support, pages 17–24, Cham. Springer
    International Publishing.
  target_id: cite-351fa121db60ef2f
  identification_status: resolved
  roles:
  - related
  decision_status: classified
  reason: '関連研究（Section 2）において、円Hough変換や複数フレームのトラックレット、および部分遮蔽に対応した2段階検出を用いる従来のサッカーボール検出手法として言及されている。

    Identification: OpenAlex referenced_works: unique_title_year_first_author_match'
  evidence:
  - Section 2 (S2.p2.1, S2.p3.1, S2.p4.1)
  - Bibliography (bib.bibx9)
  lookup:
    title: Video-based soccer ball detection in difficult situations
    year: 2015
    authors:
    - J. Halbinger
    - J. Metzler
    links: []
- ref_id: ref-10
  citation: Hariharan, B., Arbeláez, P., Girshick, R., and Malik, J. (2015). Hypercolumns
    for object segmentation and fine-grained localization. In Proceedings of the IEEE
    conference on computer vision and pattern recognition, pages 447–456.
  target_id: cite-c59f40635f1c7562
  identification_status: resolved
  roles:
  - component
  decision_status: classified
  reason: '異なる階層の畳み込み特徴マップをアップサンプリングして結合するハイパーカラム（hypercolumn）概念をDeepBallのネットワーク構成に採用している。

    Identification: OpenAlex referenced_works: unique_title_year_first_author_match'
  evidence:
  - Section 3 (S3.p1.1)
  - Section 3 Network architecture (S3.SS0.SSS0.Px1.p2.1)
  - Bibliography (bib.bibx10)
  lookup:
    title: Hypercolumns for object segmentation and fine-grained localization
    year: 2015
    authors:
    - B. Hariharan
    - P. Arbeláez
    - R. Girshick
    - J. Malik
    links: []
- ref_id: ref-11
  citation: Higham, D., Kelley, J., Hudson, C., and Goodwill, S. R. (2016). Finding
    the optimal background subtraction algorithm for eurohockey 2015 video. Procedia
    Engineering, 147:637 – 642. The Engineering of SPORT 11.
  target_id: cite-9b7165f66d627e6e
  identification_status: resolved
  roles:
  - related
  decision_status: classified
  reason: '関連研究（Section 2）において、連続フレーム差分や直前フレームの平均・中央値との差分に基づく高速な動きベース背景分割手法の先行研究として言及されている。

    Identification: OpenAlex referenced_works: unique_title_year_first_author_match'
  evidence:
  - Section 2 (S2.p1.1)
  - Bibliography (bib.bibx11)
  lookup:
    title: Finding the optimal background subtraction algorithm for eurohockey 2015
      video
    year: 2016
    authors:
    - D. Higham
    - J. Kelley
    - C. Hudson
    - S. R. Goodwill
    links: []
- ref_id: ref-12
  citation: Kia, M. (2016). Ball automatic detection and tracking in long shot views.
    International Journal of Computer Science and Network Security (IJCSNS), 16(6):1.
  target_id: null
  identification_status: unresolved
  roles:
  - related
  decision_status: classified
  reason: '関連研究（Section 2）において、色特徴に基づく背景差分を用いる従来のボール検出・追跡手法として言及されている。

    Identification: OpenAlex referenced_works (24 candidates): no_title_match'
  evidence:
  - Section 2 (S2.p1.1)
  - Bibliography (bib.bibx12)
  lookup:
    title: Ball automatic detection and tracking in long shot views
    year: 2016
    authors:
    - M. Kia
    links: []
- ref_id: ref-13
  citation: 'Kingma, D. P. and Ba, J. (2014). Adam: A method for stochastic optimization.
    arXiv preprint arXiv:1412.6980.'
  target_id: cite-3f510c73510772c0
  identification_status: resolved
  roles:
  - component
  decision_status: classified
  reason: 'DeepBallネットワークの学習における最適化手法としてAdamオプティマイザを採用している。

    Identification: explicit external ID in reference'
  evidence:
  - Section 3 Training dataset (S3.SS0.SSS0.Px3.p3.1)
  - Bibliography (bib.bibx13)
- ref_id: ref-14
  citation: Leo, M., D’Orazio, T., Spagnolo, P., Mazzeo, P. L., and Distante, A. (2008).
    Sift based ball recognition in soccer images. In International Conference on Image
    and Signal Processing, pages 263–272. Springer.
  target_id: cite-164712773859c622
  identification_status: resolved
  roles:
  - related
  decision_status: classified
  reason: '関連研究（Section 2）において、動きベースの背景差分、変形円Hough変換、および2段階候補抽出・検証を用いる従来のサッカーボール検出手法として言及されている。

    Identification: OpenAlex referenced_works: unique_title_year_first_author_match'
  evidence:
  - Section 2 (S2.p1.1, S2.p2.1)
  - Bibliography (bib.bibx14)
  lookup:
    title: Sift based ball recognition in soccer images
    year: 2008
    authors:
    - M. Leo
    - T. D’Orazio
    - P. Spagnolo
    - P. L. Mazzeo
    - A. Distante
    links: []
- ref_id: ref-15
  citation: Lin, T.-Y., Goyal, P., Girshick, R., He, K., and Dollár, P. (2017). Focal
    loss for dense object detection. arXiv preprint arXiv:1708.02002.
  target_id: cite-6303a6185aebf062
  identification_status: resolved
  roles:
  - related
  decision_status: classified
  reason: '関連研究（Section 2）において、1段階の物体検出器（RetinaNet）の代表例として言及されている。

    Identification: explicit external ID in reference'
  evidence:
  - Section 2 (S2.p5.1)
  - Bibliography (bib.bibx15)
- ref_id: ref-16
  citation: 'Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C.-Y., and
    Berg, A. C. (2016). Ssd: Single shot multibox detector. In European conference
    on computer vision, pages 21–37. Springer.'
  target_id: null
  identification_status: unresolved
  roles:
  - component
  - related
  decision_status: classified
  reason: '関連研究で1段階物体検出器（SSD）として紹介するとともに、提案手法の損失関数（分類交差エントロピー損失）およびハードネガティブマイニング戦略の基盤要素として採用している。

    Identification: OpenAlex referenced_works (24 candidates): year_mismatch_or_missing'
  evidence:
  - Section 2 (S2.p5.1)
  - Section 3 (S3.p1.1)
  - Section 3 Loss function (S3.SS0.SSS0.Px2.p1.1, S3.SS0.SSS0.Px2.p3.1)
  - Bibliography (bib.bibx16)
  lookup:
    title: 'Ssd: Single shot multibox detector'
    year: 2016
    authors:
    - W. Liu
    - D. Anguelov
    - D. Erhan
    - C. Szegedy
    - S. Reed
    - C.-Y. Fu
    - A. C. Berg
    links: []
- ref_id: ref-17
  citation: Mazzeo, P. L., Leo, M., Spagnolo, P., and Nitti, M. (2012). Soccer ball
    detection by comparing different feature extraction methodologies. Advances in
    Artificial Intelligence, 2012:6.
  target_id: cite-f9dd9f420614dba7
  identification_status: resolved
  roles:
  - related
  decision_status: classified
  reason: '関連研究（Section 2）において、動きベースの背景差分と2段階候補抽出・検証を用いる従来のサッカーボール検出手法として言及されている。

    Identification: OpenAlex referenced_works: unique_title_year_first_author_match'
  evidence:
  - Section 2 (S2.p1.1, S2.p2.1)
  - Bibliography (bib.bibx17)
  lookup:
    title: Soccer ball detection by comparing different feature extraction methodologies
    year: 2012
    authors:
    - P. L. Mazzeo
    - M. Leo
    - P. Spagnolo
    - M. Nitti
    links: []
- ref_id: ref-18
  citation: Pallavi, V., Mukherjee, J., Majumdar, A. K., and Sural, S. (2008). Ball
    detection from broadcast soccer videos using static and dynamic features. Journal
    of Visual Communication and Image Representation, 19(7):426 – 436.
  target_id: cite-f6c065fe7031aeab
  identification_status: resolved
  roles:
  - related
  decision_status: classified
  reason: '関連研究（Section 2）において、連続フレームのボール候補から有向重み付きグラフを構築して最長パスから軌跡を求める従来のボール検出手法として言及されている。

    Identification: OpenAlex referenced_works: unique_title_year_first_author_match'
  evidence:
  - Section 2 (S2.p3.1)
  - Bibliography (bib.bibx18)
  lookup:
    title: Ball detection from broadcast soccer videos using static and dynamic features
    year: 2008
    authors:
    - V. Pallavi
    - J. Mukherjee
    - A. K. Majumdar
    - S. Sural
    links: []
- ref_id: ref-19
  citation: Paszke, A., Gross, S., Chintala, S., Chanan, G., Yang, E., DeVito, Z.,
    Lin, Z., Desmaison, A., Antiga, L., and Lerer, A. (2017). Automatic differentiation
    in pytorch.
  target_id: null
  identification_status: unresolved
  roles: []
  decision_status: unclassifiable
  reason: 'Section 4で評価対象の全手法を実装・実行するためのソフトウェアフレームワーク（PyTorch）としてのみ言及されており、手法の基盤・要素・比較対象・関連研究・ベンチマークのいずれの役割にも該当しないため分類不能とした。

    Identification: OpenAlex referenced_works (24 candidates): no_title_match'
  evidence:
  - Section 4 Evaluation results (S4.SS0.SSS0.Px3.p1.1)
  - Bibliography (bib.bibx19)
  lookup:
    title: Automatic differentiation in pytorch
    year: 2017
    authors:
    - A. Paszke
    - S. Gross
    - S. Chintala
    - G. Chanan
    - E. Yang
    - Z. DeVito
    - Z. Lin
    - A. Desmaison
    - L. Antiga
    - A. Lerer
    links: []
- ref_id: ref-20
  citation: Poppe, C., De Bruyne, S., Verstockt, S., and Van de Walle, R. (2010).
    Multi-camera analysis of soccer sequences. In Advanced Video and Signal Based
    Surveillance (AVSS), 2010 Seventh IEEE International Conference on, pages 26–31.
    IEEE.
  target_id: cite-2801ae1292b2e3f2
  identification_status: resolved
  roles:
  - related
  decision_status: classified
  reason: '関連研究（Section 2）において、変形円Hough変換を用いてボール候補ブロブを検証する従来のサッカーボール検出手法として言及されている。

    Identification: OpenAlex referenced_works: unique_title_year_first_author_match'
  evidence:
  - Section 2 (S2.p2.1)
  - Bibliography (bib.bibx20)
  lookup:
    title: Multi-camera analysis of soccer sequences
    year: 2010
    authors:
    - C. Poppe
    - S. De Bruyne
    - S. Verstockt
    - R. Van de Walle
    links: []
- ref_id: ref-21
  citation: Rao, U. and Pati, U. C. (2015). A novel algorithm for detection of soccer
    ball and player. In Communications and Signal Processing (ICCSP), 2015 International
    Conference on, pages 0344–0348. IEEE.
  target_id: null
  identification_status: unresolved
  roles:
  - related
  decision_status: classified
  reason: '関連研究（Section 2）において、Aliら（2012）と同様に前景からピッチ白線を除去する従来のサッカーボール検出手法として言及されている。

    Identification: OpenAlex referenced_works (24 candidates): author_mismatch_or_missing'
  evidence:
  - Section 2 (S2.p3.1)
  - Bibliography (bib.bibx21)
  lookup:
    title: A novel algorithm for detection of soccer ball and player
    year: 2015
    authors:
    - U. Rao
    - U. C. Pati
    links: []
- ref_id: ref-22
  citation: 'Redmon, J., Divvala, S., Girshick, R., and Farhadi, A. (2016). You only
    look once: Unified, real-time object detection. In Proceedings of the IEEE conference
    on computer vision and pattern recognition, pages 779–788.'
  target_id: cite-ca7fe67fc71c45e7
  identification_status: resolved
  roles:
  - related
  decision_status: classified
  reason: '関連研究（Section 2）で1段階物体検出器（YOLO）の先行研究として紹介され、Section 3では単一パス検出器の着想元として言及されている。

    Identification: OpenAlex referenced_works: unique_title_year_first_author_match'
  evidence:
  - Section 2 (S2.p5.1)
  - Section 3 (S3.p1.1)
  - Bibliography (bib.bibx22)
  lookup:
    title: 'You only look once: Unified, real-time object detection'
    year: 2016
    authors:
    - J. Redmon
    - S. Divvala
    - R. Girshick
    - A. Farhadi
    links: []
- ref_id: ref-23
  citation: 'Ren, S., He, K., Girshick, R., and Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks. In Advances in neural
    information processing systems, pages 91–99.'
  target_id: null
  identification_status: unresolved
  roles:
  - related
  decision_status: classified
  reason: '関連研究（Section 2）において、2段階の物体検出器（Faster R-CNN）の代表例として言及されている。

    Identification: OpenAlex referenced_works (24 candidates): year_mismatch_or_missing'
  evidence:
  - Section 2 (S2.p5.1)
  - Bibliography (bib.bibx23)
  lookup:
    title: 'Faster r-cnn: Towards real-time object detection with region proposal
      networks'
    year: 2015
    authors:
    - S. Ren
    - K. He
    - R. Girshick
    - J. Sun
    links: []
- ref_id: ref-24
  citation: Reno, V., Mosca, N., Marani, R., Nitti, M., D’Orazio, T., and Stella,
    E. (2018). Convolutional neural networks based ball detection in tennis games.
    In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition
    Workshops, pages 1758–1764.
  target_id: cite-8f0c4c4fb0a47be0
  identification_status: resolved
  roles:
  - comparison
  - related
  decision_status: classified
  reason: '関連研究（Section 2）でスライディングウィンドウ方式のCNNボール検出器として議論され、Section 4およびTable 2で同一データセット上の比較対象として評価されている。

    Identification: OpenAlex referenced_works: unique_title_year_first_author_match'
  evidence:
  - Section 2 (S2.p7.1)
  - Section 4 Evaluation results (S4.SS0.SSS0.Px3.p2.1, S4.SS0.SSS0.Px3.p4.1, S4.SS0.SSS0.Px3.p5.1)
  - Table 2 (S4.T2.1.6)
  - Section 5 (S5.p1.1)
  - Bibliography (bib.bibx24)
  lookup:
    title: Convolutional neural networks based ball detection in tennis games
    year: 2018
    authors:
    - V. Reno
    - N. Mosca
    - R. Marani
    - M. Nitti
    - T. D’Orazio
    - E. Stella
    links: []
- ref_id: ref-25
  citation: 'Speck, D., Barros, P., Weber, C., and Wermter, S. (2017). Ball localization
    for robocup soccer using convolutional neural networks. In Behnke, S., Sheh, R.,
    Sarıel, S., and Lee, D. D., editors, RoboCup 2016: Robot World Cup XX, pages 19–30,
    Cham. Springer International Publishing.'
  target_id: cite-26aaef7c634ef283
  identification_status: resolved
  roles:
  - comparison
  - related
  decision_status: classified
  reason: '関連研究（Section 2）でCNNによるボール位置推定手法として議論され、Section 4およびTable 2で同一データセット上の比較対象（Model
    1 soft-sign）として評価されている。

    Identification: OpenAlex referenced_works: unique_title_year_first_author_match'
  evidence:
  - Section 2 (S2.p6.1)
  - Section 4 Evaluation results (S4.SS0.SSS0.Px3.p2.1, S4.SS0.SSS0.Px3.p3.1, S4.SS0.SSS0.Px3.p5.1)
  - Table 2 (S4.T2.1.5)
  - Section 5 (S5.p1.1)
  - Bibliography (bib.bibx25)
  lookup:
    title: Ball localization for robocup soccer using convolutional neural networks
    year: 2017
    authors:
    - D. Speck
    - P. Barros
    - C. Weber
    - S. Wermter
    links: []
- ref_id: ref-26
  citation: Yuen, H., Princen, J., Illingworth, J., and Kittler, J. (1990). Comparative
    study of hough transform methods for circle finding. Image and vision computing,
    8(1):71–77.
  target_id: null
  identification_status: unresolved
  roles:
  - related
  decision_status: classified
  reason: '関連研究（Section 2）において、従来のボール検出手法が候補ブロブの検証に利用する円Hough変換（Circle Hough Transform）の関連文献として言及されている。

    Identification: OpenAlex referenced_works (24 candidates): no_title_match'
  evidence:
  - Section 2 (S2.p2.1)
  - Bibliography (bib.bibx26)
  lookup:
    title: Comparative study of hough transform methods for circle finding
    year: 1990
    authors:
    - H. Yuen
    - J. Princen
    - J. Illingworth
    - J. Kittler
    links: []
schema_version: 1
---
Reader: agy 1.5.9, gemini-4-hydrogen-high (observed: Gemini 4 Hydrogen (High)). Reading elapsed: 99.276 seconds. Input: html, https://arxiv.org/html/1902.07304.

Legacy identification results invalidated; original reading and classifications retained.
