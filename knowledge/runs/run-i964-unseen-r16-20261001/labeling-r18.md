# run18: 予測を開く前の部分参照ラベル

凍結3edefd77とrun17追記に従い、保存されたperson_tracking v5の全79 raw track・38,132実観測boxを対象にした。予測のidentity/association/selection配列、prediction.jsonの内容、prediction overlay動画は開いていない。completeの完了状態と動画のffprobe情報、ファイルhashだけを先に確認した。推論は追加していない。

`tests/benchmarks/person_unseen_labels.py`はplan、person-execute、raw person_tracking artifactとsource RGBだけを読む。全trackを最大12等間隔crop（最初/最後の実観測を含む）にし、16枚のoverviewをすべて目視した。全景9枚（video_000 cam0の0/677、cam1の1354、cam2の677、video_001全cameraの803、video_002 cam0/2の373）で、主コートの選手2人と隣コート人物の区別、camera間の服装・位置の対応を確認した。予測IDをラベルに転記していない。

`blind-label-details-r18-requests.json`の5/10/15 frame間隔の追加crop15枚を確認した。video_000 cam0の隣コート交差、video_001 cam0の遠方人物/選手handoff・cam1背景box、video_002 cam0の隣コートとcam2の静止したフェンス誤検出を調べた。`blind-label-boundaries-r18-requests.json`の6枚では、video_000 cam0 r4/r5の人物切替を1frame間隔で確認し、r8/r11の重なりとr10の部分box、video_001 cam0 r19の白スカート人物の部分boxを確認した。全ての生成画像を見たという意味ではなく、上記46枚を目視した記録である。

- video_000 cam0 r4: 白帽子の人物から暗色の人物へ変わる。最後の白帽子観測f104、次の暗色観測f113（間は非観測）なので、reviewの境界はf105。f80–103は2人を含む曖昧boxでnull。
- 同r5: 最初は暗色人物、f56–67は2人の重なり、次の観測f76は白帽子人物。f56–75をnull、f76以後を白帽子人物とした。
- 同r11: f230–253は2人を含みnull、f254以降は白帽子人物。
- video_001 cam1 r4/r5: フェンスと小さい上部像で人物を一意に判別できず、全15 boxをnull。cam2の固定位置の白い点/縦構造は非人物誤検出Fとして明示した。

全cameraで未レビューtrack・区間欠落・未知人物を既存materialize関数で拒否した。38,132 boxのうち71はnull。GSIの1,089補間boxは実観測に含めず、boxは既存schemaどおり0.1pxに丸める。datasetの各clip `annotations/player_association/{labels.json,review.yaml}`が保存先で、git内labels-r18は今回のimmutable証拠snapshot。[全パスとhash](label-receipt-r18.json)を正本とする。

これはCodexによるRGB目視注釈で、別の人間による二重注釈は未実施。全raw trackを見たが、38,132 boxを1枚ずつ独立に目視していない。小さい遠方像、遮蔽、短い未抽出の切替を見落とす可能性がある。今回の推論自身のboxを参照にするためbox被覆は自己参照の上限に近く、未知人物や未検出frameを補わない。完全な検出recallでも独立検出器のGTでもない。既存devは旧観測box由来なので絶対値の直接比較にはこの参照差がある。同じ収録/選手に属する未見clipであり、新人物・新会場への汎化ではない。

2つの新規合成テストで、人物切替の区間反映、実観測だけの保持、未レビューtrack拒否、予測ファイル参照禁止、既存ラベル上書き拒否を確認した。初回テストはfixtureの括弧不足で収集失敗（0実行）、修正後2件成功（pytest -n4）、ruff/mypy成功。人物src/config・既存採点関数の変更は0。このラベルと手順をcommit/pushしてから一回採点へ進む。
