[PR #2182](https://github.com/Motoki0705/tennis-lab/pull/2182)へ最新mainを取り込み、確認結果を追記しました。

HEAD: `b6801e760367097a77973aaa5a2806fee1b24f5ba42b`

- FP32/FP16・DINO拡張pinとModelSpec、cache、runtime、A〜Eの接続をCPUで確認。
- 追加6ケースを含むCPU **1,484件成功**、pre-commit成功。
- [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44330974434)は`Python`を含む全14ジョブ、[Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44330974497)は全2ジョブ成功。

GPU・再レビューは未実施です。PRはReady、`MERGEABLE / CLEAN`。merge・closeは親へ返します。