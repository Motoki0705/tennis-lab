# シーンの外観変換とNHT再学習

B00の既存SfMを再利用し、画像の外観だけを変更した派生シーンを作る。
画像生成はOpenAI Images APIの編集エンドポイントを使用する。
実装はこのディレクトリ、入口は`../scripts/run_appearance_variant.py`、設定は
`../configs/run_appearance_variant.yaml`。

## B00 / Flare の設定

- 対象区間：`frame_000000.jpg`〜`frame_000248.jpg`（両端込み249枚）。
- 抽出：`round(i * 248 / 49)`による50枚。元の分割を保って学習42枚・評価8枚。
- 参照：`frame_000124.jpg`から生成し、採用後は全変換で同じ画像を使う。
- APIモデル：`gpt-image-2.5-flare-2026-09-08`、品質`high`。同条件比較後にFlareを採用した。
  実APIで未対応と確認した`input_fidelity`は送信しない。
- APIへの入力・出力：1536×864 PNG。APIの16画素単位制約と元画像の16:9を両立する。
- NHTへの入力：原寸1920×1080 JPGへ戻し、そこから960×540 PNGを生成する。
- NHT：7,000ステップ。元のseed 42、カメラ、SfM、factor 2、`pose_opt=false`を維持する。

正式な仕様は[Sunburstモデル](https://developers.openai.com/api/docs/models/gpt-image-2.5-sunburst)と
[Images編集API](https://developers.openai.com/api/reference/python/resources/images/methods/edit)を参照。
この実装は既存の`requests`依存を使用し、Codexの組み込み画像生成機能を呼ばない。

## APIキー

`OPENAI_API_KEY`環境変数、または`OPENAI_API_KEY_FILE`で指定したGit管理外の
ファイルにキーを設定する。ファイルの既定値は`~/.config/tennis-lab/openai.env`。
今回使用したWindowsの`C:\Users\kamim\.codex\openai.env`は、WSLでは
`OPENAI_API_KEY_FILE=/mnt/c/Users/kamim/.codex/openai.env`で指定できる。

```dotenv
OPENAI_API_KEY=取得したAPIキー
```

実行環境に`OPENAI_API_KEY`があればそちらを優先する。キーは設定スナップショット、
リクエスト記録、ログへ保存しない。認証ファイルをシェルスクリプトとして実行しない。
キーが空なら通信前に停止する。`check_api`はローカル検査のみで課金を伴わない。
既存variantのキー保存先だけを変える場合は`action=configure_api_key`を使用する。
モデル・プロンプト・画像は変更せず、保存先の変更履歴だけを残す。

## Sunburst / Flare の比較

```bash
.venv/bin/python -m src.synthetic_data_generation.scripts.run_appearance_variant action=compare_models
```

`comparison.reference`と`comparison.target_index`を入力とし、2モデルを同じ品質・寸法・
プロンプト・入力順序で並行して呼び出す。比較用の前処理済み画像は両モデルで同じファイルを使う。
`comparison.output_root`へ要求・応答メタデータ・PNG原本・4面の`comparison.png`を保存する。
所要時間はリクエスト単位の実測値であり、1回の比較から一般的な速度差は断定しない。
比較の再実行では既存の結果を再利用する。異なる条件で試す場合は出力ディレクトリを変える。
初回比較の`v001`は未対応パラメータによるHTTP 400を記録し、修正後は`v002`として保存する。

## 保存先

デフォルトは実行checkout内の`data/synthetic_data_generation/scene_variants/B00/clay-flare-v001/`。
`APPEARANCE_DATA_ROOT`で`data/synthetic_data_generation`に相当する絶対パスを変更できる。
以前の組み込みimagegenの試行`clay-v001/`とは独立している。

```text
clay-flare-v001/
├── variant.yaml / manifest.json       # 固定設定、選択、ハッシュ、進捗
├── inputs/
│   ├── targets/                      # 原画像50枚の実コピー
│   ├── reference-source.jpg
│   └── generation/                   # APIへ送る1536×864 PNG
├── prompts/{reference,transfer}.txt   # 実際の固定プロンプト
├── reference/clay.png                 # 採用した固定参照
├── generation/
│   ├── requests.jsonl
│   ├── attempts/<request-id>/         # 要求、APIメタデータ、生成原本、レビュー
│   └── accepted/                     # NHT取込用の原寸画像
├── provenance/
│   ├── source/                       # 元設定とメタデータ
│   └── code/                         # 学習投入時の3リポジトリの版・差分・未追跡ファイル
├── nht-config.yaml                   # finalizeで作るNHT公開設定
├── reconstruction/
│   ├── frames/{images,training-images}/
│   ├── frames/frames.json
│   ├── sfm/model/                    # rigs/framesを含めSfM全体をコピー
│   ├── sfm/reconstruction.json
│   ├── import-provenance.json / run.json / resolved-config.yaml
│   └── 3dgs/ / export/ / logs/
└── review/                           # 原画像・生成画像・50%重ね合わせ
```

元B00への書き込み、元画像とのハードリンク、未変換画像による穴埋めは行わない。
元画像の画質測定値はsource provenanceに保持し、生成画像の測定値として扱わない。

## 実行

専用worktreeのルートで実行する。NHTソースは実行checkoutのsubmoduleを使用する。
共有データと共有trainer環境を使う場合の設定例:

```bash
cd /home/kamimura/projects/tennis-lab/.claude/worktrees/b00-clay-variant
export APPEARANCE_DATA_ROOT=/home/kamimura/projects/tennis-lab/data/synthetic_data_generation
export NHT_TRAINING_PYTHON=/home/kamimura/projects/tennis-lab/third_party/nht/.trainer-venv/bin/python
export OPENAI_API_KEY_FILE=/mnt/c/Users/kamim/.codex/openai.env
.venv/bin/python -m src.synthetic_data_generation.scripts.run_appearance_variant action=prepare
.venv/bin/python -m src.synthetic_data_generation.scripts.run_appearance_variant action=check_api
.venv/bin/python -m src.synthetic_data_generation.scripts.run_appearance_variant action=generate
```

最初は参照画像を1枚生成し、`request_id`と保存先`path`を返す。
生成と目視での採用判定を分けている。コードは画像寸法やハッシュを検証するが、
幾何が保持されたという目視判定を自動で捏造しない。

```bash
.venv/bin/python -m src.synthetic_data_generation.scripts.run_appearance_variant \
  action=record_result \
  result.request_id=reference-r01-01 \
  result.path=/home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00/clay-flare-v001/generation/attempts/reference-r01-01/api-result.png \
  result.accepted=true \
  'result.notes=クレーの色調と既存の構造を確認した'
```

参照採用後の`generate`は順に`000000`、`000121`、`000248`の3視点を返す。
以後も`generate → 確認 → record_result`を繰り返す。Codexがレビューを進める場合も
同じCLIを使用でき、ユーザーがフレームごとのプロンプトを組み立てる必要はない。
不採用は`result.accepted=false`と理由を記録する。各画像は最大3試行。
50枚すべてを採用すると`ready`になる。

固定参照が決まった後は、生成をまとめて実行できる。

```bash
.venv/bin/python -m src.synthetic_data_generation.scripts.run_appearance_variant action=generate_batch
```

既定は2並列・送信開始間隔13秒。生成済み画像は再利用し、未採用の画像だけを対象にする。
`batch.indices=[121,248]`で固定選択内の画像に限定できる。API出力は先に保存し、
`next_request → record_result`による目視レビューは生成後に行う。API成功だけで採用扱いにしない。
進捗は`generation/batch/status.json`、一覧は`review/batch-*.jpg`に保存する。

今回のFlare runでは比較用の固定参照を`batch.import_reference()`で実コピーし、
`reference/import.json`に由来を記録した。比較の1枚目は`batch.reuse_comparison()`で
モデル・プロンプト・両入力のハッシュ一致を検証して再利用したため、新規API生成は49枚。
再利用した応答には元のAPI request IDを保ち、`new_api_call=false`を記録する。

```bash
.venv/bin/python -m src.synthetic_data_generation.scripts.run_appearance_variant action=finalize
.venv/bin/python -m src.synthetic_data_generation.scripts.run_appearance_variant action=train
```

`finalize`はNHTの公開CLIでSfMと画像を取り込み、実際に縮小画像を作り直す。
`train`は元repoの共有`.training_queue/`へ`resource=all`で登録する。
`execute_training`の直接実行は拒否する。完了時にNHT標準exportとcheckpointを保存する。
実験後の知見はプロジェクトのknowledge-control手順で登録する。

## 枚数・学習ステップを変える派生実験

`action=derive`は採用済みの親variantから、別の保存先へ参照と生成結果を実コピーする。
APIパラメータ・プロンプト・入力ハッシュを照合して親のレビューを継承し、
学習用JPEGも親と同じハッシュになることを検証する。checkpointは引き継がず、
派生先でゼロから学習する。親と選択規則は`provenance/derived-from.json`に保存する。

```bash
B00_VARIANTS_ROOT=/home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scene_variants/B00
.venv/bin/python -m src.synthetic_data_generation.scripts.run_appearance_variant \
  action=derive variant.output_root="$B00_VARIANTS_ROOT/clay-flare-50-30k-v001" \
  variant.scene_id=B00-clay-flare-50-30k-v001 variant.max_steps=30000
.venv/bin/python -m src.synthetic_data_generation.scripts.run_appearance_variant \
  action=derive variant.output_root="$B00_VARIANTS_ROOT/clay-flare-100-30k-v001" \
  variant.scene_id=B00-clay-flare-100-30k-v001 variant.sample_count=100 variant.max_steps=30000
```

親は`derive.parent_root`で指定する。既定は最初の50枚のFlare variant。
追加視点は既存視点間の最大間隔を二分して選び、同じ間隔なら早い側を優先する。
選択は`variant.frame_indices`に明示し、既存50枚が100枚版から脱落しない。
100枚版は元の分割規則により学習86枚・評価14枚になる。モデル比較には
両方に含まれる評価8枚の指標も別途比較する。

派生後は対象の`variant.output_root`を指定して従来の`generate_batch`、
`next_request`、`record_result`、`finalize`、`train`を使用する。
50枚版は全画像を再利用できるため、そのまま`finalize → train`へ進める。
100枚版の追加50枚のAPI生成は、50枚版のGPU学習と並行できる。

学習完了後の`action=report`は`report.roots`の共通評価画像を名前で照合し、
評価用入力の画素ハッシュが同じことを確認して指標と比較画像を保存する。
出力先は`report.output_root`。指標はNHT標準評価の値であり、trainerの既存動作上、
保存checkpointから1回optimizerを更新した後のモデルに対する評価である。
`reporting.write_training_curves()`でNHTのTensorBoardタグに対応した損失・Gaussian数の図を作る。

## 再開・エラー

- `prepare`の再実行は同一設定と元ファイルのハッシュを検証する。
- 未レビューの`generate`を再実行しても、保存済みAPI結果を返して再課金を避ける。
- 単発生成・一括生成・比較結果の再利用は同じ試行ディレクトリのロックを共有し、
  同じリクエストの同時送信や保存済み応答の上書きを防ぐ。
- API失敗・通信切断・プロセス中断は記録して停止する。自動のモデル変更・再送をしない。
  失敗内容を確認して再送する場合は`action=generate api_retry=true`を明示する。
  応答不明の中断では、再送が追加課金になり得る。
- 成功応答はAPI request ID、使用量（返された場合）、パラメータ、所要時間、画像ハッシュを保存する。
- API出力の寸法不一致は不採用になる。切り抜きや余白追加で合わせない。
- 固定参照・プロンプト・入力・採用画像の変更は検出して停止する。
- 学習ジョブが失敗した場合は原因を解消してから`action=finalize`で入力を再検証し、
  `action=train training_retry=true`で再投入する。前回のqueue記録は保存され、
  待機中・実行中のジョブは重複投入しない。
- 強制中断でmanifestが`training`のままでも、共有queueが処理終了を確認して
  `failed`/`cancelled`になっていれば`action=train training_retry=true`で再検証・復旧できる。
  元manifestとqueue記録を保存してから再投入する。ホスト再起動などでqueue自体が
  `running`のまま残った場合は自動復旧せず、queueの終了確認・復旧を先に行う。
- SfM再計算や画像選択の変更は派生workspaceで拒否する。別variantを作る。
- 「幾何保持」は入力画像とカメラの保持であり、学習中のGaussian位置を固定する意味ではない。
  生成AIの画素単位の同一性は保証しないため、生成原本を保存して学習入力を再利用する。

## 検証

```bash
.venv/bin/python -m pytest -n 0 tests/unit/synthetic_data_generation/appearance
NHT_SOURCE_WORKSPACE=/home/kamimura/projects/tennis-lab/data/synthetic_data_generation/scenes/B00/reconstruction \
  .venv/bin/python -m pytest -n 0 tests/integration/synthetic_data_generation/test_appearance_import.py
```

API単体テストはHTTPを置き換え、入力順序・モデル固定・秘密値の非出力・再開・失敗処理を検証する。
実データのCPU統合テストは、NHTの公開取込CLIから50枚のみのParser読み込みまで通し、
元の全491カメラで計算した正規化・scene scaleと42/8分割の一致を検証する。
API呼び出しとGPU学習は通常のテストでは実行しない。
実際に行った生成・学習の結果は`knowledge/nodes/`のB00 clay run群に記録している。
