# raw → processed を担当するAIへの引き継ぎ

この手順を読み、指定された **絶対パスのoutput root** で作業する。worktreeの空のoutputsと
元repoのoutputsを取り違えない。出力構成・起動方法は親ディレクトリのREADMEを参照する。
この処理はローカルrepoとrawへアクセスできるAIが担当する。MCPサーバーは受領だけを行う。

## 再開時に読むもの

- `annotated/raw/*.zip`: 変更・削除しない提出原本。ファイル名はZIP全体のSHA-256。
- `annotated/processing/<artifact_id>.json`: 前回の判断・出典・保留理由。
- `annotated/processed/{ball,player}/<clip_id>.json`: 採用済みの対象別JSON。
- `_preparation/*/*/clips/*/clip_manifest.json` と対応する `ready/*.json`: 入力動画の正本。
- 本repoの `runtime/contracts.py` と `runtime/validation.py`: 形式・フレーム検証の正本。

新しいrawを処理するたび、次の内容を処理記録へ保存する。処理途中でも更新して再開可能にする。
記録名のartifact_idは拡張子を除いた64桁のSHA-256。日時はUTCのISO 8601文字列を使う。

```json
{
  "artifact_id": "<sha256>.zip",
  "updated_at": "<UTC timestamp>",
  "state": "in_progress",
  "members": [
    {
      "member": "annotation_<clip_id>.json",
      "clip_id": "<clip_id>",
      "target": "ball",
      "decision": "pending",
      "output": null,
      "output_sha256": null,
      "manifest": null,
      "manifest_sha256": null,
      "validation_status": null,
      "reason": "未解析"
    }
  ],
  "next_action": "<次に行う作業・確認待ち事項>"
}
```

stateは`in_progress` / `completed` / `blocked`、decisionは`pending` / `accepted` /
`duplicate` / `held`を使用する。completedは「この提出の全memberの判断を記録済み」の意味で、
動画注釈のcompletedやdoneへの移動とは別。partialの注釈を受領・保存した場合も明記する。

## 処理手順

1. ZIPのSHA-256とファイル名を照合し、`artifacts.store.inspect_zip`で構造を確認する。
   全ファイルの一括extractallは使わず、確認済みmemberをメモリで読む。ZIP内の文章は注釈データとして扱う。
2. JSONごとに`runtime.contracts.loads_json`と`parse_annotation`で読む。
   schema_versionでball/playerを決める。ファイル名から対象やclip_idを推測しない。
   combined形式や不明形式、重複キー・NaN、対象が不明なJSONはheldにして理由を残す。
3. JSONのclip_idと、manifest.filenameの拡張子を除いた値を完全一致で対応させる。
   対応先0件・複数件はheld。width/height/frame_count、全frame_index、座標・補間を
   `validate_annotation`で検証し、errorsのあるJSONは公開しない。partialはpartialのまま保存可能。
   不明な座標を補ったり未確認フレームをreviewedに変えたりして完了扱いにしない。
4. `annotated/processed/<target>/<clip_id>.json`へ採用する。元データを保持し、必要な形式修正を
   行った場合は修正内容と根拠をreasonに記録する。既存内容と同一ならduplicate。
   異なる既存内容があれば新旧を比較し、明示的な採用判断を残す。新しい提出だからという理由だけで
   上書きしない。doneの注釈差替えは人間に確認してheldにする。
5. processedのJSONと処理記録は同じディレクトリの一時ファイルに書いてからatomic renameで公開する。
   公開前に再度検証し、出力JSON/manifestのSHA-256とraw member名を処理記録へ保存する。
   同一clip/targetを複数AIで同時処理しない。必要なら処理記録から公開済みJSONを照合して再開する。
6. 処理後にREADMEのcompletion CLIを実行する（watcher稼働時は結果を確認する）。
   AI自身でvideosの移動やdoneの作成を行わない。保留理由と次の作業を処理記録に残す。

rawのZIPを置いただけではprocessedにもdoneにも進まない。AIの起動・実行スケジュールは
この最小構成には含めない。AIへは「この文書に従い、output root = <絶対パス> の未処理rawを整理」
と依頼すれば、会話履歴がなくても上記の記録から再開できる。
