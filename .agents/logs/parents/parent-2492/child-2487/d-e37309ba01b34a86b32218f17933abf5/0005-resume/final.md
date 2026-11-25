RF-DETR-L の速度計測6件（PT32・PT16・TRT16 × batch 1・2）は、11月25日 08:51 JST に実行前に取り消されました。取消理由は未確認です。速度・ピークVRAM・出力一致度は未計測で、現行DINOとの速度比較も保留です。[#2487 の証拠](https://github.com/Motoki0705/tennis-lab/issues/2487#issuecomment-7148014359)

取消レシート6件と固定96画像のmanifestをSHA-256で照合し、`codex/parent-2492-issue-2487-rfdetr` の `68d57da729491069c4dcc1207aa5dd7b5f6e9b98` に記録しました。既存テスト27件とknowledge検証は通過しました。疑似GTに対する既測のRecall@0.50は、PT32・PT16・TRT16の順に96.81%・96.82%・96.81%です。

再開には、CUDA同期と出力照合の共通規約を反映した #2497 のHEADと、6条件の再投入可否を親から指定してください。[#2497 のレビュー](https://github.com/Motoki0705/tennis-lab/pull/2497#issuecomment-7138771687)

PARENT_RESULT {"schema_version":1,"state":"needs_parent","wait":null}