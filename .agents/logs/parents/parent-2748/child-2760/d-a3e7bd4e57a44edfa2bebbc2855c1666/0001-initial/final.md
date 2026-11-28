S4 を実装し、`Closes #2760` 付きの ready PR を作成しました。[PR #2768](https://github.com/Motoki0705/tennis-lab/pull/2768)

- ブランチ `codex/parent-2748-issue-2760-primary`
- HEAD `a6b35817a1a39491632fc3fa106be3406fad470f`

frozen V4 の69クリップ・330打球で、打点高327件（99.1%）、後球速324件（98.2%）、前球速232件（70.3%）、主ネット通過275件（83.3%）、実着地226件（68.5%）を抽出しました。全295交差のうち2件はネットスパン外でした。着地欠測の内訳と train/val のビン分布は PR に記録しました。[監査 JSON]([local path])

ユニット・回帰158件、PostgreSQL 17.12 結合9件、pre-commit の Ruff・mypy が合格しました。解析弾道での絶対誤差は打点高と着地位置で各0.027 mm以内、前後球速で0.0062 m/s以内でした。検証用 Compose は停止済みです。

```bash
.venv/bin/python -m pytest -n 0 tests/unit/tennis_scene/analytics
CI_REQUIRE_POSTGRES=1 .venv/bin/python -m pytest -n 0 -m postgres tests/integration/tennis_scene/analytics
.venv/bin/python -m src.tennis_scene.analytics geometry --benchmark data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v4
.venv/bin/pre-commit run
```

ライン余裕は球の投影円とライン外縁で計算し、DB には既存制約に沿ってクリップごとに1つのプロファイルを保存します。S1 の全69クリップ投入からの通し検証は #2758 合流後に残ります。PR CI は起動済みです。

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}