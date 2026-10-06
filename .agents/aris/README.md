# ARIS Codex for Tennis Lab

ARISのCodex版から、調査・比較実験・分析・レポートに必要な25スキルと依存資産を
プロジェクト専用で導入している。配布元・固定commit・スキル一覧は
[upstream.toml](upstream.toml)、利用条件は[LICENSE](LICENSE)を参照。
GPU実行・レビュー・記録の適応方針は[INTEGRATION.md](INTEGRATION.md)が正本。

## 使用

このcheckoutをCodexで開き、初回・worktree移動後に実行する。

```bash
.venv/bin/python .agents/aris/aris.py setup
.venv/bin/python .agents/aris/aris.py doctor
```

次のターンから `$research-pipeline` を指定できる。例:

```text
$research-pipeline
SfMの比較実験を進める。
対象はB00〜B03、比較手法は既存COLMAPとVidMap。
GPU実行の合計上限は6時間、まず短い区間で動作確認する。
既存の評価条件を固定し、日本語の比較レポートを作成する。
```

上記は呼び出し例であり、導入によって実行される実験ではない。
計画がある場合は `$experiment-bridge`、保存結果の比較は `$analyze-results`、
レポートのHTML化は `$render-html` を直接指定できる。
スキルが表示されない場合はこのcheckoutで新しいターン/セッションを開始する。

## 配置と更新

`.agents/skills/`にスキルとshared referencesを置き、このディレクトリには
必要な上流helper・template・LICENSEと小さな実行adapterを置く。
`aris.py setup`はcheckout固有の補助パスだけを`.aris/`へ作る。
インストール時に個人設定・外部サービス・GPU workerは変更しない。

公式インストーラーの絶対symlinkに代えてcopy配置を採用した。
`skill-installer`の`install-skill-from-github.py`で固定commitのCodexスキルを
取得し、`tools/skill-groups.tsv`の依存を含めた。特許・投稿・クラウド調達・通知などの
別用途はこの研究用セットに含めていない。

更新時は専用worktreeで新しい上流commitを選び、別の一時ディレクトリへ取得する。
`upstream.toml`の一覧と`tools/skill-groups.tsv`から依存差分を確認し、現在のコピーと
比較して取り込む。既存宛先を拒否するinstallerの動作を迂回して上書きしない。
各SKILLのintegration入口、ローカル実行/レビュー用の差替え、相対リンク、以下の検証を
維持してから`upstream.toml`を更新する。upstream updaterの一括上書きは使わない。

```bash
.venv/bin/python .agents/aris/aris.py setup
.venv/bin/python .agents/aris/aris.py doctor
.venv/bin/python -m pytest -n 0 tests/e2e/skills/test_aris_integration.py
```

テストは隔離git repositoryとCPUコマンドを使い、共有実験queue・GPU・モデルAPIを
利用しない。スキルの科学的判断や実際のSfM手法の精度は、この導入検証の対象外。
