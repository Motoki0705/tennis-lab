# #964 run 6 — ROI前の人物source比較（固定4 dev clip）

推薦・理由・リスク・残るwide欠落の解釈は[knowledge node](../../nodes/person_tracking/000005-run-i964-fullframe-selection-r6-20260930.md)を参照。暫定推薦はCOCO全画面 .30、既定は未変更。

参照boxは旧COCO/旧trackerから作った部分ラベルで、**COCOに有利**。旧boxとの一致を検出recallとは呼ばない。未ラベル予測をFPと数えない。全sourceは800/1333・全画面・ROI前。同じmotion/IoU BoT-SORTを使うsource比較で、最終tracking方式比較ではない。

## 固定した選別規則

box下端中心を校正cameraのz=0平面へ投影。シングルス幅 |x|≤4.115m、|y|≤16.885mのcoreにおけるdistinctな実観測frameがclipの25%以上となる連結groupを選手候補とし、その後に上限6。ダブルス幅 |x|≤5.485mは診断値で、出力を切らない。**選択済み断片の全実観測を保持する**（wide run・無効足元も含む）。

gap>1秒または既存0.25秒窓/3m jumpで区間を切る。両方coreへ入る断片を位置連続性＋CLIP cosine≥.8のvetoで連結。mutual nearestと次点差.2を要求、1秒未満の曖昧断片は除外、同camera handoff重複は≤.2秒。滞在の重複は1回だけ数える。選別maskは重複観測も保持し、既存v3対応に渡すgroup timelineだけ早いraw track IDで1box/frameへ正規化する。検出sourceの閾値以降にscore gate/fusionは無い。ラベルはこの判定へ渡さない。

全raw trackでcropの遮蔽を検査し、coreに入るtrackへ既定CLIPをCPUで適用。同じ動画/hash・重み/hash・frame・resized RGBが完全一致するcropだけ再利用する。crop不足・短いsegmentに埋め込みが無い場合はmissingと記録し、幾何だけで照合する既存挙動。camera間対応は同じ既存CLIP+幾何の第2確認で、undecidedを補完しない。

主表は選別直後、IoU≥.3。IoU≥.5、clip別、identity別、対応成功clipだけのassociated表はCSV/JSONに保存。選手ID保持は当該集計範囲でラベルunitの50%以上、非選手ID完全除外は残存0。unit=(clip,camera,frame,person)で重複boxをまとめる。near/farは同frameの2選手のbox下端順位、順位が定まらなければunknown。

## 全体

|source|選手ID保持|選手unit保持|非選手ID全除外|非選手unit除外（全体）|非選手unit除外（追跡hit内）|隣コート除外|コート外除外|wide保持|対応決定clip|
|---|---|---|---|---|---|---|---|---|---|
|ft_base_0.01|8/8|18323/20558|7/8|3818/4454|2221/2857|395/395|1844/2480|141/143|3/4|
|ft_base_0.02|8/8|18021/20558|7/8|4125/4454|1336/1665|395/395|2151/2480|135/143|4/4|
|ft_base_0.05|8/8|17499/20558|7/8|4432/4454|932/954|395/395|2458/2480|121/143|3/4|
|coco_fullframe_0.05|7/8|14280/20558|7/8|4452/4454|4448/4450|395/395|2478/2480|49/143|4/4|
|coco_fullframe_0.10|8/8|18045/20558|8/8|4454/4454|4453/4453|395/395|2480/2480|111/143|3/4|
|coco_fullframe_0.30|8/8|19843/20558|7/8|4451/4454|4451/4454|395/395|2477/2480|123/143|4/4|
|union_fullframe_0.30|8/8|19787/20558|7/8|4446/4454|4441/4449|395/395|2472/2480|143/143|4/4|

## camera × near/far

|source|camera|近遠|選手ID保持|選手unit保持|非選手unit除外（全体）|非選手unit除外（追跡hit内）|隣コート除外|コート外除外|wide保持|
|---|---|---|---|---|---|---|---|---|---|
|ft_base_0.01|cam0|near|4/4|3149/3270|561/561|561/561|381/381|180/180|0/0|
|ft_base_0.01|cam0|far|4/4|2538/3270|279/897|66/684|0/0|279/897|49/49|
|ft_base_0.01|cam0|unknown|4/4|227/227|58/76|58/76|14/14|44/62|0/0|
|ft_base_0.01|cam1|near|4/4|3427/3430|78/78|78/78|0/0|78/78|0/0|
|ft_base_0.01|cam1|far|4/4|2699/3430|1146/1146|762/762|0/0|1146/1146|71/71|
|ft_base_0.01|cam1|unknown|3/3|67/67|13/13|13/13|0/0|13/13|0/0|
|ft_base_0.01|cam2|near|4/4|2802/3367|0/0|0/0|0/0|0/0|21/23|
|ft_base_0.01|cam2|far|4/4|3285/3367|1553/1553|652/652|0/0|71/71|0/0|
|ft_base_0.01|cam2|unknown|3/4|129/130|130/130|31/31|0/0|33/33|0/0|
|ft_base_0.02|cam0|near|4/4|3158/3270|561/561|561/561|381/381|180/180|0/0|
|ft_base_0.02|cam0|far|3/4|1598/3270|606/897|126/417|0/0|606/897|47/49|
|ft_base_0.02|cam0|unknown|4/4|226/227|75/76|73/74|14/14|61/62|0/0|
|ft_base_0.02|cam1|near|4/4|3430/3430|78/78|78/78|0/0|78/78|0/0|
|ft_base_0.02|cam1|far|4/4|2882/3430|1120/1146|213/239|0/0|1120/1146|71/71|
|ft_base_0.02|cam1|unknown|3/3|67/67|2/13|2/13|0/0|2/13|0/0|
|ft_base_0.02|cam2|near|4/4|3166/3367|0/0|0/0|0/0|0/0|17/23|
|ft_base_0.02|cam2|far|4/4|3364/3367|1553/1553|258/258|0/0|71/71|0/0|
|ft_base_0.02|cam2|unknown|4/4|130/130|130/130|25/25|0/0|33/33|0/0|
|ft_base_0.05|cam0|near|4/4|3160/3270|561/561|557/557|381/381|180/180|0/0|
|ft_base_0.05|cam0|far|3/4|1310/3270|897/897|84/84|0/0|897/897|40/49|
|ft_base_0.05|cam0|unknown|4/4|224/227|76/76|59/59|14/14|62/62|0/0|
|ft_base_0.05|cam1|near|4/4|3430/3430|78/78|78/78|0/0|78/78|0/0|
|ft_base_0.05|cam1|far|4/4|3033/3430|1130/1146|53/69|0/0|1130/1146|71/71|
|ft_base_0.05|cam1|unknown|3/3|67/67|7/13|6/12|0/0|7/13|0/0|
|ft_base_0.05|cam2|near|4/4|2788/3367|0/0|0/0|0/0|0/0|10/23|
|ft_base_0.05|cam2|far|4/4|3358/3367|1553/1553|74/74|0/0|71/71|0/0|
|ft_base_0.05|cam2|unknown|3/4|129/130|130/130|21/21|0/0|33/33|0/0|
|coco_fullframe_0.05|cam0|near|4/4|2732/3270|561/561|560/560|381/381|180/180|0/0|
|coco_fullframe_0.05|cam0|far|2/4|1091/3270|897/897|897/897|0/0|897/897|29/49|
|coco_fullframe_0.05|cam0|unknown|4/4|219/227|76/76|75/75|14/14|62/62|0/0|
|coco_fullframe_0.05|cam1|near|4/4|3294/3430|78/78|78/78|0/0|78/78|0/0|
|coco_fullframe_0.05|cam1|far|3/4|2000/3430|1144/1146|1142/1144|0/0|1144/1146|0/71|
|coco_fullframe_0.05|cam1|unknown|3/3|62/67|13/13|13/13|0/0|13/13|0/0|
|coco_fullframe_0.05|cam2|near|4/4|2981/3367|0/0|0/0|0/0|0/0|20/23|
|coco_fullframe_0.05|cam2|far|3/4|1771/3367|1553/1553|1553/1553|0/0|71/71|0/0|
|coco_fullframe_0.05|cam2|unknown|4/4|130/130|130/130|130/130|0/0|33/33|0/0|
|coco_fullframe_0.10|cam0|near|4/4|2985/3270|561/561|561/561|381/381|180/180|0/0|
|coco_fullframe_0.10|cam0|far|3/4|2621/3270|897/897|896/896|0/0|897/897|33/49|
|coco_fullframe_0.10|cam0|unknown|4/4|227/227|76/76|76/76|14/14|62/62|0/0|
|coco_fullframe_0.10|cam1|near|4/4|3430/3430|78/78|78/78|0/0|78/78|0/0|
|coco_fullframe_0.10|cam1|far|3/4|2645/3430|1146/1146|1146/1146|0/0|1146/1146|58/71|
|coco_fullframe_0.10|cam1|unknown|3/3|67/67|13/13|13/13|0/0|13/13|0/0|
|coco_fullframe_0.10|cam2|near|4/4|3136/3367|0/0|0/0|0/0|0/0|20/23|
|coco_fullframe_0.10|cam2|far|4/4|2804/3367|1553/1553|1553/1553|0/0|71/71|0/0|
|coco_fullframe_0.10|cam2|unknown|4/4|130/130|130/130|130/130|0/0|33/33|0/0|
|coco_fullframe_0.30|cam0|near|4/4|3270/3270|561/561|561/561|381/381|180/180|0/0|
|coco_fullframe_0.30|cam0|far|4/4|3107/3270|897/897|897/897|0/0|897/897|29/49|
|coco_fullframe_0.30|cam0|unknown|4/4|227/227|76/76|76/76|14/14|62/62|0/0|
|coco_fullframe_0.30|cam1|near|4/4|3430/3430|78/78|78/78|0/0|78/78|0/0|
|coco_fullframe_0.30|cam1|far|4/4|3078/3430|1143/1146|1143/1146|0/0|1143/1146|71/71|
|coco_fullframe_0.30|cam1|unknown|3/3|67/67|13/13|13/13|0/0|13/13|0/0|
|coco_fullframe_0.30|cam2|near|4/4|3170/3367|0/0|0/0|0/0|0/0|23/23|
|coco_fullframe_0.30|cam2|far|4/4|3364/3367|1553/1553|1553/1553|0/0|71/71|0/0|
|coco_fullframe_0.30|cam2|unknown|4/4|130/130|130/130|130/130|0/0|33/33|0/0|
|union_fullframe_0.30|cam0|near|4/4|3270/3270|561/561|561/561|381/381|180/180|0/0|
|union_fullframe_0.30|cam0|far|4/4|3268/3270|897/897|896/896|0/0|897/897|49/49|
|union_fullframe_0.30|cam0|unknown|4/4|227/227|76/76|76/76|14/14|62/62|0/0|
|union_fullframe_0.30|cam1|near|4/4|3430/3430|78/78|78/78|0/0|78/78|0/0|
|union_fullframe_0.30|cam1|far|4/4|2919/3430|1141/1146|1137/1142|0/0|1141/1146|71/71|
|union_fullframe_0.30|cam1|unknown|3/3|67/67|10/13|10/13|0/0|10/13|0/0|
|union_fullframe_0.30|cam2|near|4/4|3181/3367|0/0|0/0|0/0|0/0|23/23|
|union_fullframe_0.30|cam2|far|4/4|3295/3367|1553/1553|1553/1553|0/0|71/71|0/0|
|union_fullframe_0.30|cam2|unknown|4/4|130/130|130/130|130/130|0/0|33/33|0/0|

## 人物候補・trackの負荷

personsは検出box件数（ユニークな実人物数ではない）。tracks_totalはcamera/clip内raw IDの累計。near/farを跨ぐIDは各層に現れるため層別累計を足し合わせない。frame平均の分母はcamera-frame。選別raw ID数と上限6の連結group数は別。

|source|camera|近遠|person box総数|person/frame|raw track累計|track/frame|最大track/frame|選別box/frame|
|---|---|---|---|---|---|---|---|---|
|coco_fullframe_0.05|all|all|214397|20.436|3930|19.524|43|1.412|
|coco_fullframe_0.05|cam0|far|69858|19.977|1058|19.361|37|0.316|
|coco_fullframe_0.05|cam0|near|19103|5.463|451|5.181|22|0.784|
|coco_fullframe_0.05|cam0|unknown|6035|1.726|355|1.669|37|0.084|
|coco_fullframe_0.05|cam1|far|63007|18.017|1157|17.141|37|0.574|
|coco_fullframe_0.05|cam1|near|5133|1.468|84|1.400|9|0.942|
|coco_fullframe_0.05|cam1|unknown|1401|0.401|137|0.388|34|0.022|
|coco_fullframe_0.05|cam2|far|43057|12.313|1111|11.576|26|0.589|
|coco_fullframe_0.05|cam2|near|4835|1.383|91|1.322|12|0.857|
|coco_fullframe_0.05|cam2|unknown|1968|0.563|102|0.535|20|0.067|
|coco_fullframe_0.10|all|all|68361|6.516|844|6.340|16|1.742|
|coco_fullframe_0.10|cam0|far|18297|5.232|288|5.082|14|0.751|
|coco_fullframe_0.10|cam0|near|7286|2.084|69|2.046|10|0.854|
|coco_fullframe_0.10|cam0|unknown|1724|0.493|97|0.479|16|0.102|
|coco_fullframe_0.10|cam1|far|17157|4.906|254|4.738|11|0.759|
|coco_fullframe_0.10|cam1|near|4122|1.179|20|1.166|5|0.981|
|coco_fullframe_0.10|cam1|unknown|466|0.133|45|0.131|14|0.027|
|coco_fullframe_0.10|cam2|far|14778|4.226|193|4.102|11|0.815|
|coco_fullframe_0.10|cam2|near|3818|1.092|33|1.081|7|0.897|
|coco_fullframe_0.10|cam2|unknown|713|0.204|31|0.196|10|0.041|
|coco_fullframe_0.30|all|all|40531|3.863|120|3.847|7|1.900|
|coco_fullframe_0.30|cam0|far|9978|2.853|40|2.840|5|0.888|
|coco_fullframe_0.30|cam0|near|6092|1.742|16|1.739|4|0.935|
|coco_fullframe_0.30|cam0|unknown|977|0.279|29|0.277|6|0.084|
|coco_fullframe_0.30|cam1|far|5131|1.467|16|1.457|3|0.881|
|coco_fullframe_0.30|cam1|near|3945|1.128|6|1.127|2|0.981|
|coco_fullframe_0.30|cam1|unknown|162|0.046|7|0.046|4|0.020|
|coco_fullframe_0.30|cam2|far|10297|2.945|34|2.930|6|0.964|
|coco_fullframe_0.30|cam2|near|3504|1.002|15|0.999|3|0.906|
|coco_fullframe_0.30|cam2|unknown|445|0.127|16|0.126|5|0.041|
|ft_base_0.01|all|all|57490|5.480|912|5.271|55|1.881|
|ft_base_0.01|cam0|far|12561|3.592|224|3.440|11|1.061|
|ft_base_0.01|cam0|near|10801|3.089|197|2.955|17|0.900|
|ft_base_0.01|cam0|unknown|1559|0.446|78|0.432|15|0.113|
|ft_base_0.01|cam1|far|6692|1.914|149|1.795|7|0.780|
|ft_base_0.01|cam1|near|5061|1.447|93|1.396|40|0.980|
|ft_base_0.01|cam1|unknown|220|0.063|17|0.059|6|0.021|
|ft_base_0.01|cam2|far|9261|2.648|69|2.611|7|0.939|
|ft_base_0.01|cam2|near|9121|2.608|179|2.526|51|0.803|
|ft_base_0.01|cam2|unknown|2214|0.633|138|0.599|48|0.045|
|ft_base_0.02|all|all|38915|3.709|383|3.633|35|1.783|
|ft_base_0.02|cam0|far|7724|2.209|98|2.150|7|0.595|
|ft_base_0.02|cam0|near|6139|1.756|53|1.732|5|0.903|
|ft_base_0.02|cam0|unknown|930|0.266|42|0.261|7|0.089|
|ft_base_0.02|cam1|far|4431|1.267|53|1.230|4|0.832|
|ft_base_0.02|cam1|near|4111|1.176|30|1.161|21|0.981|
|ft_base_0.02|cam1|unknown|170|0.049|11|0.048|4|0.032|
|ft_base_0.02|cam2|far|8022|2.294|29|2.282|5|0.962|
|ft_base_0.02|cam2|near|6073|1.737|115|1.691|33|0.907|
|ft_base_0.02|cam2|unknown|1315|0.376|96|0.346|34|0.049|
|ft_base_0.05|all|all|30520|2.909|206|2.876|21|1.678|
|ft_base_0.05|cam0|far|4733|1.353|59|1.323|4|0.376|
|ft_base_0.05|cam0|near|5419|1.550|20|1.545|3|0.904|
|ft_base_0.05|cam0|unknown|677|0.194|25|0.192|5|0.071|
|ft_base_0.05|cam1|far|3708|1.060|22|1.049|3|0.876|
|ft_base_0.05|cam1|near|3980|1.138|15|1.133|11|0.981|
|ft_base_0.05|cam1|unknown|157|0.045|9|0.045|3|0.030|
|ft_base_0.05|cam2|far|6803|1.945|25|1.936|4|0.960|
|ft_base_0.05|cam2|near|4266|1.220|60|1.200|19|0.797|
|ft_base_0.05|cam2|unknown|777|0.222|57|0.204|20|0.038|
|union_fullframe_0.30|all|all|40917|3.900|138|3.882|7|1.895|
|union_fullframe_0.30|cam0|far|10129|2.896|44|2.883|5|0.935|
|union_fullframe_0.30|cam0|near|6140|1.756|23|1.750|4|0.936|
|union_fullframe_0.30|cam0|unknown|1041|0.298|33|0.295|6|0.079|
|union_fullframe_0.30|cam1|far|5196|1.486|18|1.474|3|0.836|
|union_fullframe_0.30|cam1|near|3944|1.128|6|1.127|2|0.981|
|union_fullframe_0.30|cam1|unknown|163|0.047|8|0.046|4|0.020|
|union_fullframe_0.30|cam2|far|10318|2.951|40|2.937|6|0.942|
|union_fullframe_0.30|cam2|near|3504|1.002|15|1.000|3|0.910|
|union_fullframe_0.30|cam2|unknown|482|0.138|19|0.134|7|0.046|

## 既存CLIPのcamera間対応

|source|clip|状態|理由|camera別候補group|pair F1 IoU .3|.5|
|---|---|---|---|---|---|---|
|ft_base_0.01|video_000/clip_000|undecided|ambiguous_players|{'cam0': 2, 'cam1': 3, 'cam2': 2}|—|—|
|ft_base_0.01|video_000/clip_007|ok||{'cam0': 2, 'cam1': 2, 'cam2': 2}|0.8993|0.8960|
|ft_base_0.01|video_001/clip_001|ok||{'cam0': 4, 'cam1': 2, 'cam2': 2}|0.8132|0.8365|
|ft_base_0.01|video_002/clip_013|ok||{'cam0': 3, 'cam1': 2, 'cam2': 2}|0.9381|0.9476|
|ft_base_0.02|video_000/clip_000|ok||{'cam0': 2, 'cam1': 2, 'cam2': 2}|0.9148|0.9218|
|ft_base_0.02|video_000/clip_007|ok||{'cam0': 2, 'cam1': 2, 'cam2': 2}|0.9987|0.9990|
|ft_base_0.02|video_001/clip_001|ok||{'cam0': 2, 'cam1': 2, 'cam2': 2}|0.7718|0.7963|
|ft_base_0.02|video_002/clip_013|ok||{'cam0': 2, 'cam1': 2, 'cam2': 2}|0.9699|0.9725|
|ft_base_0.05|video_000/clip_000|undecided|ambiguous_players|{'cam0': 2, 'cam1': 3, 'cam2': 2}|—|—|
|ft_base_0.05|video_000/clip_007|ok||{'cam0': 2, 'cam1': 2, 'cam2': 2}|0.9445|0.9433|
|ft_base_0.05|video_001/clip_001|ok||{'cam0': 1, 'cam1': 2, 'cam2': 2}|0.8528|0.8575|
|ft_base_0.05|video_002/clip_013|ok||{'cam0': 2, 'cam1': 2, 'cam2': 2}|0.9518|0.9588|
|coco_fullframe_0.05|video_000/clip_000|ok||{'cam0': 3, 'cam1': 2, 'cam2': 1}|0.4524|0.4534|
|coco_fullframe_0.05|video_000/clip_007|ok||{'cam0': 2, 'cam1': 2, 'cam2': 3}|0.5740|0.5760|
|coco_fullframe_0.05|video_001/clip_001|ok||{'cam0': 1, 'cam1': 2, 'cam2': 4}|0.4729|0.4732|
|coco_fullframe_0.05|video_002/clip_013|ok||{'cam0': 3, 'cam1': 2, 'cam2': 3}|0.6385|0.6406|
|coco_fullframe_0.10|video_000/clip_000|ok||{'cam0': 2, 'cam1': 3, 'cam2': 2}|0.8458|0.8458|
|coco_fullframe_0.10|video_000/clip_007|undecided|ambiguous_association|{'cam0': 2, 'cam1': 2, 'cam2': 3}|—|—|
|coco_fullframe_0.10|video_001/clip_001|ok||{'cam0': 3, 'cam1': 2, 'cam2': 2}|0.8388|0.8388|
|coco_fullframe_0.10|video_002/clip_013|ok||{'cam0': 3, 'cam1': 3, 'cam2': 2}|0.9814|0.9814|
|coco_fullframe_0.30|video_000/clip_000|ok||{'cam0': 2, 'cam1': 2, 'cam2': 2}|1.0000|1.0000|
|coco_fullframe_0.30|video_000/clip_007|ok||{'cam0': 2, 'cam1': 2, 'cam2': 3}|0.9454|0.9454|
|coco_fullframe_0.30|video_001/clip_001|ok||{'cam0': 2, 'cam1': 2, 'cam2': 2}|0.9287|0.9287|
|coco_fullframe_0.30|video_002/clip_013|ok||{'cam0': 3, 'cam1': 2, 'cam2': 2}|0.9889|0.9889|
|union_fullframe_0.30|video_000/clip_000|ok||{'cam0': 3, 'cam1': 2, 'cam2': 2}|0.9733|0.9736|
|union_fullframe_0.30|video_000/clip_007|ok||{'cam0': 2, 'cam1': 2, 'cam2': 2}|0.9703|0.9708|
|union_fullframe_0.30|video_001/clip_001|ok||{'cam0': 3, 'cam1': 2, 'cam2': 2}|0.9285|0.9287|
|union_fullframe_0.30|video_002/clip_013|ok||{'cam0': 2, 'cam1': 2, 'cam2': 2}|0.9651|0.9652|

pair F1は既存#933指標で、追跡boxに照合できたunit間だけの条件付き指標。未追跡の選手unitは主表の保持分母には残るがpair F1には入らない。値が高くても観測取りこぼしや未決定clipを無視して良い意味ではない。

成功clipだけの対応指標を全4clipの主表と同一視しない。sideは既存の注釈ballによる判定を固定した。最終方式・encoder比較、未見、全pipeline完走は未実施。pipeline既定・#937重み/閾値は変更していない。
