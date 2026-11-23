# 部分フレームの人物対応注釈

保存済み経路から疑義 frame と確信層の無作為標本を選び、同期 RGB 上の全検出 box に匿名選手 ID、`non_player`、`ambiguous` を割り当てる。meiji_4cam の保存入力の組み立て方と実測は [試行記録](../../../../knowledge/runs/run-i2442-meiji-pseudo-association-pilot/README.md) を参照。

## 抽出

[設定](../configs/pseudo_annotation_meiji.yaml)は source frame、メートル、秒、保存解像度の px を単位にする。カメラごとのローカルコート座標で絶対座標を比較するため、コートの向きが未確定でも保存校正があるカメラの足元を使える。

| 層 | 保存成果物から拾う条件 |
| --- | --- |
| 選別外れ・コート近傍 | 元 track の実観測が選別から外れ、box 下端中央を逆投影した足元が指定領域内 |
| 1視点・3D欠測 | 同じ選手の実観測の割当が1カメラにあり、同じ frame の有効な3D関節が0 |
| 未割当・再投影不一致 | 選別 group の実観測が最終 ID 未割当、または同一選手の指定数以上の関節で保存残差が閾値超過・再投影理由で棄却 |
| ID switch・断片 | コート近傍の track で足元の中央値が急変、短い実観測区間、または近傍で途切れて短い間隔で復帰 |
| 停止 clip | 設定にある理由で止まった clip のうち、どれかのカメラに検出がある frame |
| 確信層 | 全選手が指定以上の割当 view 数と有効3D関節数を持ち、上のどの疑義にも該当しない frame |

`extract` は全候補と試算用の抽出済み frame を層別に保存する。疑義層は clip の0 frame から揃えた固定幅の区間を seed 付きハッシュで並べ、上位の区間の該当 frame を取る。確信層は frame 単位で選ぶ。1 frame が複数層で選ばれた場合も依頼は1回。対応が停止した clip は、停止理由・暫定診断の有無と、保存済みの前段から得た層を記録する。

## 実行と採点

```bash
.venv/bin/python -m src.tasks.player_association.pseudo_annotation extract "$snapshot" "$selection"
.venv/bin/python -m src.tasks.player_association.pseudo_annotation prepare "$snapshot" "$selection" video_000/clip_032 "$packet" --stratum single_view_no_3d
.venv/bin/python -m src.tasks.player_association.pseudo_annotation run "$packet" "$result"
.venv/bin/python -m src.tasks.player_association.pseudo_annotation publish "$packet" "$result" "$dataset"
```

`prepare` の一件は1 clip の半開区間に収まる source frame 集合。対象 frame ごとに、同期した全カメラの全景と番号付き crop を1枚にする。box は保存済み `person_detection` v2 の順序。既定は RGB と候補番号だけを提示する。`--show-prior-association` は実観測の source row に結び付く raw ID と旧選手 ID を提示する試験用の選択肢。

`run` は [作業指示書](terra_instructions.md)を使い、`gpt-6.1-terra`、`xhigh` で実行する。同じ dataset・clip を同時に実行できるのは1件、同一ホスト・OSユーザーの上限は4件。使用量と経過時間は各 `receipt.json`、CLI のイベントは `events.jsonl` に保存する。再試行は別の実行ディレクトリを指定する。各 clip の公開は、対象 frame 全体を含む一件の採用結果で行う。

出力は既存の [`player_association_labels_v1`](../evaluation/labels.py)。形式検査で各 frame・カメラ・box の過不足と入力 SHA-256 の不一致を拒否する。箱座標は固定入力から写し、選手名は clip 内の P1〜P4（シングルスは P1〜P2）。`non_player` は box 単位、曖昧は v1 の `null`。`provenance.pseudo_annotation` に版、注釈済み frame（box が0件の frame を含む）、層、モデル、effort、プロンプトと入力の SHA-256 を保持する。公開時は同じ同期 RGB と clip manifest を照合して既存の dataset 注釈パスに書く。

人物対応・追跡の採点は明示された frame 集合を使う。時間方向の switch 判定は連続する注釈区間の内側で行う。抜き取りでは [review ツール](../review/README.md)で注釈済み frame の一覧から移動し、全カメラの box と crop を照合できる。
