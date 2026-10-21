# Git履歴書き換えの対応表

2026-10-21に #1310（親 #1267）の本番候補を検証した記録。
GitHubへの履歴反映と、この参照更新branchの公開は親セッションが行う。

| 対象 | SHA |
|---|---|
| 旧main | `088628519e3dfe2fe5e522469fff277097465e6f` |
| 書き換え後main | `262b7d8520f0991da0e8215c16975155a69f4fe0` |
| 両mainのtree（完全一致） | `d013ee764ff0335b1cb02edf4f5d1be88f0d8cb3` |

mainの 9,376 blob / 484,532,905 bytesを直接比較し一致した。
modeと4つのsubmodule pointerも一致している。
main内容検査のSHA-256は `e419e96fd149c4861df884e364b72c0aa18615efbe251db2661ea70167876626`
（パス・mode・blobのbytesをパス順で連結したdigest）。

[commit-map.txt](commit-map.txt) はgit-filter-repo 2.47.0の出力をそのまま保存したもの。
SHA-256: `288c591916aec045473c8da2a5d49df2b26f55d7a3eb53afcf331a41f697f716`。
5,348 commit（5,318変更、消滅0）を対応付ける。
空commit・mergeと、commit message内の由来SHAは保持した。署名は保持されない。

## 除去対象と検証

- knowledge: 3,027パス。HEAD削除記録1,024パスと、履歴だけの2,003パス。
  パス別最大versionの合計 1,521,246,186 bytes、対象のunique blobは
  2,621個 / 1,433,539,830 bytes。
- 実際に消えたknowledge blobは 2,610個 /
  1,419,854,989 bytes。対象外の `paper/` には同内容の
  11 blob / 13,684,841 bytesが残る（うち5 blobは現在のmainで使用）。
  パスの除去と、同じ内容の全面的な消去は異なる。
- SMPL派生資産: 承認された5 blob / 1,345,862 bytesだけをIDで除去した。
  mainでは元から使われていない。[全パス・commit数・別版](smpl-blob-history.md)。
- metrics.tex・probe.cpp・Dockerfile・uncommitted.patch 2件は保持。
  その他の現存mainファイル、再現用patch、削除manifest、研究内容・数値は保持した。

全387 branch・11 tagの先端を、元treeから対象パスと承認5 blobのエントリだけを
引いた期待値と比較した。全 5,348 commitについても、元のtreeから
独立に計算したGit tree hashと一致した。mainは除外なしで同一。

`--strip-blobs-with-ids` 単独では、対象blobへの変更が落ちると以前の許可版 `asset` が
親から継承される場合があった。元commitで承認blobを使うエントリへ明示的な削除を加え、
許可版は元から存在した3 commitだけで保持した。追加の除去blobはない。

## 通常commitとしての参照更新

2,615箇所 / 1,601ファイルの参照を更新した。
固定run例外13件のhashだけを更新し、例外集合を増やしていない。
既存のpatch・削除記録など668ファイルはbyte比較で同一。
公開したこのディレクトリは、再実行時に旧SHAの由来を壊さないよう置換から保護する。

既存ツール2 commitはGitHubのadvertised historyに含まれていなかったため、raw commit-mapへ
架空の行は足していない。新mainの上へpatchを順に適用し、stable patch-idの一致を確認した。

| 元のツールcommit | 新main上のcommit |
|---|---|
| `8d92424e16bcbc6ca276799f5e3ee8c1bbebcb87` | `332129b843f2e2a8a9e558389d0e4dadc6bca74c` |
| `0f6870139bd2ef4c60c2a586d0b951ac6740fce6` | `68459c4a048e7f5e5104432eda8cf56c1a17a63a` |

この2 commitに続く参照更新commitで、この対応表と説明を保存する。
同commitにはdirty checkout拒否条件のruff修正と、このディレクトリの保護テストも含む。

検証: knowledge・構造境界のCPUテスト149件合格、pre-commitのruff/mypy、全tree検証とfsck。
`kg_validate.py --check-summary --base-ref main` は前後とも0 errors / 218 warnings。
`kg_repro_paths.py --json` は前後とも476 OK・missing 0・unverifiable 17で、対象別のstatusも一致。
[未解決候補とノードの一覧](unresolved-references.md) は解決したものとして扱わない。

HF model cardの旧再現commit `9c74f881b0e52a759f1e3ad1cefe2aa0096981ab` は
`8433ba70d09ce6f26fb7dea15340903c19237b9d` に対応する。5箇所の更新案を用意した。
issue本文の更新案は188本文・554箇所。両者のオンライン反映は親がレビューして行う。
モデルのHF revision・重みSHA-256・外部repoのSHAは変えない。

## 容量と取得上の限界

| コピー（apparent bytes、checkout込み） | 前 | 後 |
|---|---:|---:|
| 通常clone | 2,014,052,725 | 1,063,450,941 |
| shallow clone | 837,553,138 | 837,552,481 |
| 全refのmirror | 1,736,217,099 | 762,138,161 |
| 旧PR refsを保持する模擬mirror | 1,736,217,099 | 1,727,198,232 |

submodule本体・データ・重み・仮想環境は含めない。独立したfile:// cloneで測った。
GitHubの `refs/pull/*` は更新できず、PR参照やキャッシュから旧commitを取得できる場合がある。
GitHub側の容量の同等縮小や、旧blobの完全消去は保証しない。
[GitHubの説明](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)。

mapにある新SHAでも、PR refだけに接続しているものは通常cloneから到達できない。
branch/tagからの到達性を確認して使う。旧patch・削除manifestの由来SHAは更新していない。
旧cloneでpull/mergeして新旧履歴を接続せず、新cloneへ未公開差分だけを移す。
