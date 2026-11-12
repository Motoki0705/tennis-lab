# #1913 アブレーション結果の証拠調査（planning: 調査と報告のみ）

親issue: https://github.com/Motoki0705/tennis-lab/issues/1913 （本文を最初に全文読むこと）

## あなたの役割

親（Claude）がknowledgeを自分で記録するための**一次証拠の収集と照合**を行う。
knowledgeノードの作成、PRの作成・クローズ・コメント、issueへの書込み、Driveへの書込み・移動・削除、学習・評価ジョブの起動は**一切しない**。
repoの追跡ファイルは変更しない。成果物はすべて `tmp/investigation-1913/` 配下（gitignore済み）に置く。

Driveの読み取りは `.agents/skills/tennis-drive/SKILL.md`、Colabの状態確認（読み取りのみ）は `.agents/skills/tennis-colab/SKILL.md` に従う。
knowledgeの形式は `knowledge/README.md` と `.agents/skills/knowledge-control/` を参照する（書かないが、親が記録できる形で証拠を揃える）。

## 調査項目

各項目で、観測値は「実行条件・評価split・seed・更新数・由来（Drive file ID/パス、SHA-256）」と一緒に記録する。
issue本文の値をそのまま写さず、Driveの一次JSONを取得して照合する。照合できなかったものは「未照合」と明記する。

1. **既存のknowledge PRの内容調査**: PR #1805、#1831、#1910、#1489（および #1897 が追加する `knowledge/runs/run-ball-arch-efficiency-s42-20261111/` 配下）。
   - 各PRのknowledge差分（ノードID・連番・frontmatterの config/metrics/artifacts・本文の主張・runs配下のファイル）を一覧化する。
   - 主張・数値をDriveの一次証拠と照合し、誤り・古い記述・現状との差（例: #1805は6kまでで12kを含まない）を列挙する。
   - 各PRに knowledge 以外の変更（src・CI・再現スクリプト等）があるか、PRをクローズした場合に失われるものは何か、依存PR（#1858→#1855→#1831、#1892→#1896→#1897→#1910）との関係を整理する。
   - そのまま新PRへ再利用できるファイル（再現スクリプト・小さな指標JSON・receipt等）と、その出所の妥当性を判定する。
2. **#1675 ボール容量 7条件**: 共通DPT 10k＋query各6k と 累積12k の終端validation JSON 14件を取得・hash照合し、mean/median/P95/20px以内などの指標、評価frame数、seed、checkpointの所在を表にする。12層・24層が保留であることの根拠コメントも確認する。
3. **#1899 構成切り分け**: 完了2条件（ConvNeXt・4query＋mHC、X3D/CSN・1query）の終端JSONと旧#1514基準の評価identity一致の根拠。3条件目の**現在の**最新保存地点（Driveのcheckpoint状態を再取得）と、稼働中のwriter/Colab割当の有無。
4. **#1581/#1586 Court全6ペア**: 9評価JSONすべてを取得して完了印のSHA-256と照合し、ペアごとのKP誤差などの指標表を作る。#1831のgroupノードの集計値（同群57.36px、異群4.57px、差+52.80px、P03 2.97px）を再計算で検証する。
5. **#1859 構造・効率**: 未学習L4 benchmark（#1910の記録）の一次データ所在と値。T1前半6kのvalidation JSON。T1後半・T2・C1・C2・C3の**現在の**Drive保存状態（更新数、終端評価の有無）を再取得する。
6. **#1367/#1476 独立test**: Driveの同名shards 3件の中身（サイズ・hash・manifest有無）、配布完了manifestの有無、PR #1489の現状。knowledgeに記録できる確定事実と、未完了の停止点を分ける。
7. **#1231 旧hybrid・#986**: issue・コメント・PR・Drive（outputs/ball_detection/train 以外も含め、名前・日付で探索）・ローカルの `.training_queue/` や記録された出力先で原成果物を探す。見つからなければ探索範囲を列挙する。#986のclose理由と32条件の実施範囲を確認する。
8. **元issueのクローズ可否の材料**: #1675、#1899、#1581、#1586、#1859、#1367、#1476、#1514（関連するなら）について、issue本文の完了条件・最新コメントの合意/保留と、現時点で満たされた条件・満たされない条件を一覧化する。クローズの判断は親と人間が行うので、推奨は付けてよいが根拠を分けて書く。
9. **競合の確認**: mainの `knowledge/nodes/ball_detection/.sequence` と `court_detection/.sequence` の現在値、各PRが使う連番、mainのsummary.mdの該当節の現在の記述。

## 成果物

- `tmp/investigation-1913/report.md`: 項目ごとの事実・根拠リンク・SHA-256・未照合点・推奨。日本語。
- `tmp/investigation-1913/evidence/<項目>/`: 取得した小さな一次JSON（評価結果・完了印・状態ファイル）。ファイル名と元のDriveパス/IDの対応を `evidence/index.json` に書く（path, drive_id, size, sha256）。checkpointや予測配列はダウンロードしない（数MB超のものは取得しない）。
- 最終メッセージ: report.md の要約（各項目3行以内）と、親の判断が必要な点。

## 作業量と終了条件

- 作業時間の上限は**150分**。項目1〜9を順に進め、各項目は最大20分を目安にする。
- 早めに止めてよいのは次の場合だけ: (a) 全項目を完了した、(b) Drive/GitHub認証が失敗し、読み取りが継続不能になった（エラー全文を報告する）。
- 特定項目で一次証拠が見つからない場合は、その探索範囲を記録して次の項目へ進む。途中で全体を終えない。
