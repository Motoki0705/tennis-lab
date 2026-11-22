# 再現と証拠

PR #2200 / #2190 の同じbranch。数値処理は `freeze.json` のcode commitと内容一覧を正本とする。参照・許容差・採否閾値は不変。source freeze後にmeiji4由来の既存integration fixtureも確認し、その結果による調整はしていない。

`repro.sh`をbranchのrepo rootから `.venv/bin/python` で実行する。新しい評価用worktreeでは storage-tier の正規手順でrun70/73の共有outputsを参照可能にし、このrunのoutputsを新規に用意する。既存結果を使う場合は入力/設定の一致を検査してcase単位で再開する。異なるsourceや条件の結果を同じtagに書くことは拒否する。

- 入力raw: `outputs/tennis_scene/evaluate/court-acceptance-implementation-20261116/{raw,val_raw,test_raw}`。
- RGB/LINE場/比較の3条件: `outputs/tennis_scene/evaluate/court-fixed-support-rgb-20261117`。8530 raw、8530 LINE、3545 RGBについて、run73の内容inventoryと全件照合した。モデルforward・新raw取得・queueへのjob追加は0。受入比較のGPU使用は0だが、初回pytestで既存CUDA unit 1件を誤って実行した。逸脱と是正は `execution-incident.json` に記録する。
- 新しい結果: `outputs/tennis_scene/evaluate/court-semantic-support-20261117/regression`（非meiji4 3269画像）と`meiji4`（276画像）。`evaluation/<dataset>`は両者への明示的なsymlink。
- `geometry_XX.pkl`はこの実行が書いた診断用のモデル幾何で、評価scriptはpickleを読み込まない。共有入力として無検証のpickleを取り込む経路はない。
- GTは候補のHと採否を全て確定した後の採点にだけ使う。順位はaccepted/quality score/region順で、GT誤差を使わない。新RGBのfine fitは診断Hで、採用Hとは区別する。

4条件は旧処理、前回(a)、前回(a+b)、今回(a+b)。原因分解のため今回(a)単独も保存する。前回3条件は同一の保存LINE場・RGBに対するrun73の結果をそのまま使用し、今回のcodeで旧条件を再生成して履歴を上書きしない。旧3条件の時間はrun73当時の実測であり、今回と同時実行の速度比較ではない。

`design.md`、`diagnose.py`、`diagnose_scene_quality.py`に設計と原因分解を記録した。設計用の実行は`support-probe-v1`（synthetic喪失58件）、`support-scene-v1`（meiji3）、`design-v1/v2`（Court val209・synthetic99の固定subset）に残る。これらは最終評価値ではない。初回の`final-regression`はcoarse第一候補の代案検査漏れをコード確認で見つけたため31画像で中止した。結果を設計に利用せず、修正後の`regression`で全件をやり直した。いずれもmeiji4の新しい結果は見ていない。

`freeze.json`以後は数値処理を変更しない。独立性の限界: meiji4の参照とRGB精密化は同系統の白線抽出を使う。224画像は参照/pilotには未使用だが、過去の分析では全276画像を扱っており、未知のholdoutとは呼ばない。Court valのGTと既知Hの人工画像は別の証拠。syntheticの学習重複は未確認、実画像testは未定義。meiji3のGTは既存3画像だけで今回の再承認なし、tennis_clipはGTなし。16画像の追加注釈は作らない。

採点/図表は`summarize.py`、`transitions.py`、`render.py`、`figures.py`が正本。原解像度の採否・線別残差・coverage・初期必須線は`review/index.html`から全69clip×4camera×5条件を開ける。大きなJSON・画像・rawはoutputsに残し、`artifact-inventories.json`で全件の内容一覧へ接続する。

Courtモデルの新しいCUDA推論・公開経路の実機確認は未実施。共有lockが正規に復旧した後、共有training queue（half / provider codex / issue2190）で固定checkpointのforwardと公開region経路を再確認する必要がある。今回のCPU後処理時間をGPU forwardの所要時間へ読み替えない。infraの修復や迂回は行っていない。
