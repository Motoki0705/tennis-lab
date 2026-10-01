# ローカルエージェントによるボールアノテーション

ローカルのCodex CLIを1クリップずつ起動し、画像確認・注釈編集・継続・採用・再アノテーションの比較まで行う。
2026-09-30〜10-01のボール注釈キャンペーンで使用したツールを、出力ディレクトリから独立させた実装。
対象はボールのみ。選手注釈、動画の取得・分割、MCPサーバーの起動はこのモジュールの担当外。

注釈の定義とJSONの正本は [PROTOCOL](../resources/PROTOCOL.md) と
[runtime](../runtime/contracts.py)、rawからprocessedへの保存契約は
[PROCESS_RAW](../resources/PROCESS_RAW.md) を参照する。ワーカー向けの指示と、画像から判断するときの
補足・道具の使い方は [WORKER.md](WORKER.md) が唯一の保守元。

## 構成

| モジュール | 担当 |
|---|---|
| `configuration` / `common` | 明示したroot、設定検証、原子的な保存、動画検証キャッシュ |
| `campaign_state` / `dispatcher` | 状態台帳、同一clipの重複起動防止、利用枠停止・再開、並列数制御 |
| `launcher` | Pythonの監視プロセスからCodexを起動し、終了結果を保存 |
| `ct` | ワーカー用の注釈編集、一覧画像・拡大画像、候補、補間、完了報告 |
| `qa` / `phase2` | 品質指標、再アノテーションの選定、新旧比較、食い違いの確認画像 |
| `intake` | 原本保存、採用・旧維持・差し替え・再確認の指示 |
| `prefetch` | モデルを一度読み込み、動画検証とボール候補を先に計算 |
| `audit` / `efficiency` | 層別の確認画像、版ごとのトークン量・時間・品質の比較 |

すべての入口は `.venv/bin/python -m src.tennis_scene.chat_annotation.local_agent`。
`init`以外は`--campaign`をサブコマンドの前に指定する。出力先は`campaign.json`に保持し、
別rootへの暗黙の切り替えはしない。Linux/WSLの`flock`とプロセスグループを使用する。

## 1. 準備と試走

動画の準備は [親README](../README.md) の既存CLIで行う。
Codex CLIのインストール・ログインと、`view_image`による画像確認ができる環境を用意する。
起動設定の公式仕様は [非対話実行](https://learn.chatgpt.com/docs/non-interactive-mode) と
[CLIリファレンス](https://learn.chatgpt.com/docs/developer-commands?surface=cli) を参照。

```bash
ANNOTATION_ROOT=/absolute/path/to/outputs/chat_annotation
CAMPAIGN_DIR="$ANNOTATION_ROOT/local_agent"

.venv/bin/python -m src.tennis_scene.chat_annotation.local_agent \
  --campaign "$CAMPAIGN_DIR" init --root "$ANNOTATION_ROOT" \
  --model gpt-6.1-sol --effort max --parallel 1 \
  --ball-checkpoint /absolute/path/to/ball.ckpt

.venv/bin/python -m src.tennis_scene.chat_annotation.local_agent \
  --campaign "$CAMPAIGN_DIR" run --dry-run
```

`init`と`run --dry-run`はエージェントを起動しない。既存のcampaignを上書きして初期化しない。
`--ball-checkpoint`を省略すると、画像の確認と注釈編集は使えるがモデル候補の生成は使えない。
checkpointは内容のSHA-256で識別する。同名の別の重みや変更された重みの候補を再利用しない。

worktreeで使う場合も`--root`には共有したい元repo側の絶対パスを明示する。
`--project-root`と`--python`は実行するコード・環境を、`--codex-binary`と`--codex-home`は
CLI実体・認証環境を明示するための引数。`--python`ではvenvのパスを保持する。
このモジュール自身が認証キーを保存することはない。

最初は異なる撮影条件の少数クリップで、全フレーム確認・中心位置・対象外の球の除外・
出力の保存先を確認してから並列数を増やす。対象を絞るには`control.json`の`mode`を`pilot`にし、
`pilot_tasks`へ対象のtask IDを列挙する。

## 2. 注釈の実行と監視

```bash
.venv/bin/python -m src.tennis_scene.chat_annotation.local_agent \
  --campaign "$CAMPAIGN_DIR" run --exit-when-idle

# 別の端末で確認
.venv/bin/python -m src.tennis_scene.chat_annotation.local_agent \
  --campaign "$CAMPAIGN_DIR" status
```

`run`だけで起動すると継続して監視する。`--exit-when-idle`は実行可能なタスクと稼働中ワーカーが
なくなったら終了する。採用判定待ちの結果があっても、エージェントの処理は終了できる。
停止は`control.json`の`mode=drain`（新規起動を止める）または`mode=stop`（稼働中が終われば終了）で行う。
ディスパッチャだけを停止しても、既に起動したワーカーは継続する。

ワーカーは`workspace-write`サンドボックス内で、attemptディレクトリを作業rootとして動く。
ユーザー設定由来のMCP・skills・通知を読み込まない軽量起動を使用する。
ワーカーのコマンド実行は非対話・ネットワーク無効・CPUのみで、追加エージェントを起動しない。
Codex自体のAPI接続は必要。サンドボックスが使えない場合に制限なしの起動へ切り替える処理はない。

`control.json`が実効設定の正本。`max_parallel`、`max_launch_per_tick`、`adaptive`、
利用枠を残す`quota_stop_percent`、`slow_seconds`、`timeout_seconds`を設定できる。
起動待ち、確認待ち、利用枠による一時停止を区別し、設定並列数と実稼働数を混同しない。
`SLOW`は通知で、終了時刻の予測ではない。timeoutも正常完了の予定時刻ではない。

監視では`logs/events.log`の`ERROR`、`FAIL`、`QUOTA_PAUSE/RESUME`、`SLOW`、`REVIEW_BATCH`、
`IDLE`、`EXIT`を優先する。毎回すべての作業ログを読み直さず、確認待ちがまとまった時点でQAを行う。
容量不足や混雑で拒否された場合は並列数を下げて待つ。各クリップの全フレーム検証と
CPUモデル読み込みにもスロット制限がある。CUDA候補計算の実行方法は下記を参照。

## 3. 完了結果を確認して採用

ワーカーの状態は`pending → running → review / continue / failed`。
`review`は、全フレーム確認と形式検証を通った`completed`または`partial`の結果で、
親による画像確認と採用判断を待っている。意味上の正しさは形式検証では保証されない。
`continue`は注釈とNOTESを新しいattemptへ引き継ぐ。全フレーム確認済みでも未解決位置の作業は継続できる。

```bash
.venv/bin/python -m src.tennis_scene.chat_annotation.local_agent \
  --campaign "$CAMPAIGN_DIR" intake list --status review

.venv/bin/python -m src.tennis_scene.chat_annotation.local_agent \
  --campaign "$CAMPAIGN_DIR" qa task <task_id> --sheets

.venv/bin/python -m src.tennis_scene.chat_annotation.local_agent \
  --campaign "$CAMPAIGN_DIR" intake adopt --note "確認した範囲・結果・未解決点" -- <task_id>
```

親は疑わしいフレームと注釈円を確認し、判断の根拠を`--note`へ残す。
`adopt`は原本JSONをJSON-only ZIPで保存する。同じclipに別のprocessedが既にある場合は`held`にし、
比較前に上書きしない。元データ、未確認、位置不明を都合よく補正して完了扱いにしない。

修正依頼は次のように行う。指定範囲は新しいattemptで未確認に戻し、前の成果物を変更しない。

```bash
.venv/bin/python -m src.tennis_scene.chat_annotation.local_agent \
  --campaign "$CAMPAIGN_DIR" intake revise <task_id> \
  --notes "フレーム120〜125の対象球を再確認し、根拠をNOTESへ残す" --unreview 120:126
```

## 4. 再アノテーションの比較と差し替え

```bash
.venv/bin/python -m src.tennis_scene.chat_annotation.local_agent \
  --campaign "$CAMPAIGN_DIR" phase2 rank --show 30
.venv/bin/python -m src.tennis_scene.chat_annotation.local_agent \
  --campaign "$CAMPAIGN_DIR" phase2 enqueue --top 45
```

phase 2はゼロからの注釈を作り、phase 1と同じclipを同時に動かさない。
ワーカー結果をQAして`intake adopt`すると既存注釈との差がある候補は`held`になる。
その後に比較する。

```bash
.venv/bin/python -m src.tennis_scene.chat_annotation.local_agent \
  --campaign "$CAMPAIGN_DIR" phase2 compare -- <p2_task_id>
.venv/bin/python -m src.tennis_scene.chat_annotation.local_agent \
  --campaign "$CAMPAIGN_DIR" phase2 triage
.venv/bin/python -m src.tennis_scene.chat_annotation.local_agent \
  --campaign "$CAMPAIGN_DIR" phase2 disagreements -- <p2_task_id>
```

今回のキャンペーンで承認された差し替え基準を使用する。

1. 新注釈に未確認フレームがない。
2. 未解決位置の比率が旧以下。
3. 共通の位置があるフレームの中心差の中央値が2px以下。
4. 球の有無が食い違う箇所を画像で確認し、新が正しいと判断できる。

中央値・比率の閾値判定には丸め前の値を使う。中央値が未定義なら自動合格にしない。
`triage`は数値による選別で、採用はしない。短い食い違いや位置がnullの球も画像確認の対象。
保持球と対象球を切り替える基準はWORKERの補足に従い、短い区間という理由で誤ラベルを許容しない。
終了境界を確定できない場合は旧を維持するか、理由付きで再確認を依頼する。

```bash
.venv/bin/python -m src.tennis_scene.chat_annotation.local_agent \
  --campaign "$CAMPAIGN_DIR" intake replace <p2_task_id> \
  --comparison "$CAMPAIGN_DIR/qa/phase2/<clip_id>/compare.json" \
  --presence-reviewed --note "確認した全食い違い区間と、新が正しい根拠"

.venv/bin/python -m src.tennis_scene.chat_annotation.local_agent \
  --campaign "$CAMPAIGN_DIR" intake keep-old --reason "基準未達の理由" -- <p2_task_id>
```

`replace`は比較時の旧・新・manifestのSHA-256を現在の内容と照合する。
比較後に変更されていれば、比較と判断をやり直す。旧版は`annotated/history/ball/`へ不変保存し、
処理記録に公開前の意図を書いてからJSONを原子的に置換する。同じ承認内容で再実行すると、
置換の前後で停止していた処理を完了できる。別campaignも含めた採用操作は共有ロックで直列化する。

## 5. 候補の事前計算と効率確認

```bash
.venv/bin/python -m src.tennis_scene.chat_annotation.local_agent \
  --campaign "$CAMPAIGN_DIR" prefetch --device cpu --lookahead 0 --exit-when-idle
.venv/bin/python -m src.tennis_scene.chat_annotation.local_agent \
  --campaign "$CAMPAIGN_DIR" efficiency
```

候補は探索の手がかりで、ラベルではない。キャッシュを作っただけで確認済みにはならない。
同じ入力の再デコード・同じ重みの再読み込みを減らし、画像の生成と確認をまとまりごとに行う。
版ごとのリクエスト数、画像数、トークン量、所要時間を未解決比率と併せて比較する。
`efficiency`のcost値は入力1・cached入力0.1・出力8の重みを使う比較用指標で、請求額ではない。
Codexの保存済みsessionを読み、取得できた試行だけを集計する。

CUDAでの事前計算は、必ず
[training-queueの手順](../../../../.agents/skills/training-queue/SKILL.md) に従う。
queueを元repoの`.training_queue/`で共有し、half/allジョブの中で同じ`prefetch`入口へ
`--device cuda --lookahead 0 --exit-when-idle --max-minutes 120`を渡す。
入口はqueueの実行環境がないCUDA呼び出しを拒否する。ワーカー自身はGPUを使わない。

層別の確認画像を作る場合は`audit sample --seed 0 --out <output_dir>`を使い、
確認者が記入した`verdicts.json`を`audit score <output_dir>`で集計する。
この操作は画像と表を作るもので、評価用サブエージェントを自動起動しない。

## 出力と保管

```text
<annotation_root>/
  _preparation/                         # 入力の正本manifest
  videos/ または done/                  # 対応する保存済み動画
  annotated/
    raw/<sha256>.zip                     # 不変の提出原本
    processing/<sha256>.json             # 判断・出力hash・差し替えの記録
    processed/ball/<clip_id>.json         # 採用済みの正本
    history/ball/<clip_id>/<sha256>.json  # 差し替え前の旧版（campaignとは別保管）
<campaign_dir>/
  campaign.json / control.json / state.json / versions.json
  tasks/<task_id>/attempt_NN/            # task・prompt・annotation・NOTES・result・events
    work/                               # 画像・候補・中間ファイル
  cache/                                # 動画検証・モデル候補・ロック
  qa/                                   # 比較・確認画像
  logs/                                 # 実行イベント・効率集計
```

campaignのスクリプトを出力内で変更する必要はない。安定したcheckoutを使用し、稼働中はコードと
ワーカー指示を変更しない。特に実行中のシェルファイルの上書きは、元キャンペーンで重複実行を起こした。
新実装はシェル経由の再読み込みを廃止し、試行ごとに起動条件とコード版を記録する。

容量削減では稼働中ワーカーがないことを確認し、`work/`と再生成可能な`cache/`を優先する。
原本・処理記録・processed・historyは保持する。campaign全体を削除すると試行のログや
確認画像と`campaign_result`の参照先が失われるため、必要な診断資料を先に退避する。
`partial`の採用完了、campaignの終了、動画の`done/`移動は別の状態。
動画の完了移動には既存の [completion CLI](../README.md) を使用する。

## 検証

```bash
.venv/bin/python -m pytest tests/unit/tennis_scene/chat_annotation/local_agent tests/e2e/tennis_scene/test_local_agent.py
.venv/bin/python -m ruff check src/tennis_scene/chat_annotation/local_agent tests/unit/tennis_scene/chat_annotation/local_agent tests/e2e/tennis_scene/test_local_agent.py
.venv/bin/python -m mypy --follow-imports=skip src/tennis_scene/chat_annotation/local_agent
```

テストは合成動画と擬似Codex CLIを使用し、モデルAPI・実データ・GPUへアクセスしない。
原本の保存、並行採用、比較の固定、途中停止からの復旧、確認範囲の引き継ぎ、サンドボックス付き
起動引数、実際の動画デコード・画像出力を検証する。
