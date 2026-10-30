# 中央target・文脈条件のCPU調査bundle

結果・考察は[knowledge node](../../nodes/ball_detection/000062-run-ball-temporal-target-survey-20261022.md)、
当時の評価規則案は[protocol.md](protocol.md)に保持する。ここは再現手順とartifactの読み方を扱う。
**後続runにより置き換え（不採用案の記録）**。
現行規則は[全フレーム教師のprotocol](../run-ball-all-frame-protocol-20261022/protocol.md)を参照する。
このbundleは常設の窓生成APIではない。CPU集計で、RGB復号・GPU・Colab/Drive操作は行わない。

## 再現

repoの `.venv` があるactive checkout rootから実行する。`SOURCE_REPO_ROOT` は先行調査と
同じ不変snapshot・stage1 manifest・Chat原注釈を持つrepoを指す。
元manifest内のsnapshotと原注釈の絶対pathがその環境で利用できることが必要で、
可搬版の自動探索・差替えは行わない。移植は #1374 の担当。
出力はactive checkout内の新しいディレクトリでなければ拒否する。

```bash
SOURCE_REPO_ROOT=<source-repo-root> \
SURVEY_OUTPUT=tmp/ball-temporal-target-survey-repro \
bash knowledge/runs/run-ball-temporal-target-survey-20261022/repro.sh
```

`repro.sh` はmanifestを先行の `temporal-support-probe.json` と照合し、
index/metadata・clipの意味的identity・旧play窓の再生成・旧幾何計数・split間group非重複を検査する。
教師分類は `annotation_states` とhybridの空ラベル照合/不在mask APIを呼び出す。
画像payloadのopen数0と、元manifest/index/metadataおよび全1,800 shardの
inode/size/mtime/ctime不変を確認する。学習・画像復号が必要なAPIへは接続しない。
APIの内容照合値は `inputs.json` に保存する。後のsrc版で値が違う場合は結果を同一条件と呼ばず、
記録時点のAPIを持つGit版で再実行する。

記録した実行は105.86秒、最大RSS 1,084,596 KiB（約1.03 GiB）。環境依存の参考値で、
新たな実行の所要時間保証ではない。`execution.json` に記録したGit版はsrcの版を表し、
bundleのスクリプトはこのディレクトリのcommit済みファイルを使う。
再実行の `execution.json` は実時間/RSSが変わるため、再現の一致判定から除く。

## artifact

| file | 意味 |
|---|---|
| `summary.json` | split・T・文脈別の全体計数、一意targetと教師のidentity |
| `source-fps.csv` | 全T・文脈のsource別、正確なnative FPS別、source×FPS別内訳。支持数0の行も残す |
| `groups-*.csv` | 元32窓と各文脈の128/256共通候補の元動画group別計数 |
| `centers-*.jsonl` | 128/256で成立する中央ブロック。全maskブロック除外後。形式はprotocolを参照 |
| `inputs.json` | 入力版・API内容照合・判定規則・検査結果・snapshot全体のsplit/group一覧 |
| `annotation-evidence.json` | 元注釈を照合したChat clip、注釈の内容照合値、除外したnotes付き空frame数 |
| `execution.json` | CPU調査の実時間、最大RSS、src版、確認した元ファイル数 |
| `coverage.svg` | `summary.json` から生成した一意observedのcoverage図 |
| `temporal_target_probe.py` / `plot_coverage.py` / `repro.sh` | 調査と図の再現コード |
| `test_temporal_target_probe.py` | 小さな合成例による教師・境界・重複の検証 |

教師列 `observed` / `reviewed_empty` / `out_of_frame` / `masked` は幾何候補の中央領域に含まれる
一意frame数。`*_occurrences` は窓の重複を数えた出現数。
`candidate_blocks` は教師を問わない幾何候補、`supervised_blocks` は中央に1枚以上の既知教師がある数。
後者の全中央frameがidentityの対象で、`usable_masked` はその中のmask数。
snapshot行だけはclip全体のframeを幅1のblockとして数えており、中央ブロック数と混ぜない。
coverageの分母・重複owner・maskの意味はprotocolに集約した。

## 通常検証

pytestはホスト共有のロックとworker数4以下を使う。`PYTEST_LOCK` は親から指定されたpathを渡す。

```bash
PYTHONDONTWRITEBYTECODE=1 flock "$PYTEST_LOCK" .venv/bin/python -m pytest \
  knowledge/runs/run-ball-temporal-target-survey-20261022/test_temporal_target_probe.py -n 4 -q
.venv/bin/python -m ruff check knowledge/runs/run-ball-temporal-target-survey-20261022/*.py
.venv/bin/python -m mypy --follow-imports=skip knowledge/runs/run-ball-temporal-target-survey-20261022/*.py
.venv/bin/python .agents/skills/knowledge-control/scripts/kg_validate.py --check-summary --base-ref origin/main
.venv/bin/python .agents/skills/knowledge-control/scripts/kg_repro_paths.py
```

テストは対称文脈、reference/PTS障壁、重複教師の一意数と出現数、非整列tail、全maskブロック除外、
Chat notesの除外と原注釈改変の拒否を検査する。full surveyの再実行では、元データを変えずに
JSON/CSV/中心ブロック列挙が一致することも照合する。
`PYTHONDONTWRITEBYTECODE=1` はrun bundle内に検証用bytecodeを作らないために指定する。
src変更は含まないため、既存の学習経路全体やRGB復号の回帰テストはこの子の検証に含めない。

実施結果は6 tests passed、ruff/mypy成功。独立した出力先への再実行で、実時間/RSSを除く
22 artifacts（JSON/CSV/中央ブロック列挙/SVG）がbyte単位で一致した。
knowledge検査は0 errors・既存218 warnings、再現path検査は0 missing・既存8 unverifiable。
後者はこの調査以外の古いrunの参照先commitがローカルにない警告で、本bundleの未解決参照は0。
node内の全24集計行・18 source coverage値もJSON/CSVと照合した。
