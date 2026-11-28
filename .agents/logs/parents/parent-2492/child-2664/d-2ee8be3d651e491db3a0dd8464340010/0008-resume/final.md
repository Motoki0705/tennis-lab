CI で失敗した公開 API テストを修正し、RF-DETR-L・RTMPose-X の公開名4件と公開箇所を既存テストで検査するようにしました。同じ [PR #2681](https://github.com/Motoki0705/tennis-lab/pull/2681) に push 済みです。

公開 API と4モデルの CPU 単体テスト50件が5.32秒で合格し、pre-commit の ruff・mypy も通りました。新 HEAD `7d3ff104081c09994ccae93c86be1c02c22075d2` の [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45921320846) は実行中です。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2681,"head":"7d3ff104081c09994ccae93c86be1c02c22075d2","resume_on_success":false}}