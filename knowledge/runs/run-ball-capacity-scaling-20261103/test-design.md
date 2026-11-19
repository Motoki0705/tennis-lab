# テスト設計


| Invariant / bug detected | Lightest layer / reuse | Cost bound |
| --- | --- | --- |
| Irregular spatial sizes produce /8,/16,/32,/64 with T intact; radius exactly sum depths, no temporal normalization leak or accidental temporal stride | New unit test_x3d_csn.py; existing test_cnn.py::test_encoder_temporal_receptive_field_does_not_leak_through_normalization remains legacy coverage | Small configurable widths, T >=25 for radius11; <=5s/case measured |
| Chunk outputs plus input/parameter gradients match full clip at true edges; checkpointing does not change state or dependencies | New focused X3D cases reuse structure of test_execution.py::test_chunk_halos_preserve_every_feature_and_input_parameter_gradients; no legacy cross product expansion | <=4 cases, small H/W; <=5s/case target |
| Four independent queries, one mHC span gets residual-free update, CNN/cross/mHC/time/FFN/readout have gradient paths; zero-init branch gradients handled deliberately | New test_mhc_query.py; use explicit nonzero update config or update step before nonzero branch gradient assertion | <=4 cases, dim64, one/two blocks, tiny images |
| Real timestamps affect temporal attention; global time translation is invariant | Direct block/model unit with nonzero update scale and asymmetric evidence; reuse existing TimeAttention | <=2 cases |
| T32/64/128 state keys/shapes and values strict-transfer; artifact validates original T; malformed/missing fields and topology mismatches fail loudly | Unit mHC config/model I/O plus existing temporal training unit support | <=10 cases, no training |
| DPT T32 -> mHC T64 only encoder weights/moments; new transformer state empty; old T32 transfer constraints retained | Focused checkpoint unit/integration with CPU optimizer update and mismatched config | <=3 cases |
| Actual runner continuous vs interrupted/resumed state equals including weights, AdamW, RNG, pack/sample position, freeze/unfreeze, encoder moment inheritance and next update | Extend tests/integration/tasks/ball_detection/test_sequential_temporal_training.py using tests/support/tasks/ball_detection/temporal.py; retain existing legacy sequential test | One additional actual runner scenario; use 24x24 store and minimal valid dim64/depth CPU model; measure duration and record if >5s |
| Source identity includes mHC | Existing provenance test if found, otherwise direct lightweight assertion | <=1 case |

Validation planned: OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest
-n0 targeted nodes with durations; ruff and mypy changed src files. No GPU work.


親の最終確認で、評価reportのmodel_kindと不正なYAML nameの拒否を既存テストへ追加した。
