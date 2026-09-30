# 全mergeの目視監査

2026-09-30、親agentが以下の全6ページを表示して確認した（独立validator評価ではない）。
各entryはmerge frame（元keep/drop box付き）と±6 source frameの文脈crop。

|ページ|entry|観察|
|---|---|---|
|[00](merge_contacts/page_00.jpg)|0–4|同じ人物の重複box。entry2はフェンス奥の小さい人物で解像度が低い。|
|[01](merge_contacts/page_01.jpg)|5–9|同じ人物の重複box。entry9には別の人も映るが、keep/dropは同じ手前の人物を囲う。|
|[02](merge_contacts/page_02.jpg)|10–14|同じ人物の重複box。entry14の隣の人物はbox外。|
|[03](merge_contacts/page_03.jpg)|15–19|同じ人物の重複box。entry16の別人はbox外。entry17/18は左画面端で同じ身体が切れている。|
|[04](merge_contacts/page_04.jpg)|20–24|同じ人物の重複box。entry20は手前にも別人がいるが対象は遠側。entry21は屈んだ同じ人物の上体、entry22は画面右端の同じ身体の部分。|
|[05](merge_contacts/page_05.jpg)|25|同じ人物の重複box。|

keep/dropのどちらかが別の実在人物に対応する事例は認めなかった（0/26）。
小さい人物・画面外・遮蔽された身体の同一性を完全に証明するものではない。
部分参照による全26件の機械的監査でも別identity候補0件。画像から完全な全frame GTを作ったとは扱わない。

動画の全345frame読戻し後、出力frame26/145/172/303を表示して3camera上下配置、off/onタイトル、
raw/group ID・指標、mergeした元rowの表示を確認した。merge frameの保持による非一様なサンプリングはreview.jsonに記録。
