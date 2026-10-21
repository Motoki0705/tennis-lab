# SMPL承認blobの全履歴調査

全advertised ref（PR refsを含む）から到達する5,348個の異なるcommitを調べた。commit数は、そのパスでそのblobを使うcommitの数。同じcommitをrefごとに重複計上しない。

過去のパスを全raw履歴から調べ、承認blobが使われたパスと同名ファイルの全版を検査した。このmirrorではvendor移動前の別パスは見つからず、下表の5パスだった。

| path | blob | commits containing blob at path | bytes | decision |
|---|---|---:|---:|---|
| `src/submodules/vendor/gvhmr/body_model/data/smpl_coco17_J_regressor.pt` | `a95e94f6604dbfb112f01102a4ae939fa158eafc` | 3 | 5 | retain; alternate version |
| `src/submodules/vendor/gvhmr/body_model/data/smpl_coco17_J_regressor.pt` | `f9e310c1b1e0135a0bd8cce7c8a96ee679e5ffea` | 3820 | 469227 | remove |
| `src/submodules/vendor/gvhmr/body_model/data/smpl_neutral_J_regressor.pt` | `a95e94f6604dbfb112f01102a4ae939fa158eafc` | 3 | 5 | retain; alternate version |
| `src/submodules/vendor/gvhmr/body_model/data/smpl_neutral_J_regressor.pt` | `ba3f5ed90ab5f07376315c94e9675ff246efe3fb` | 3820 | 662187 | remove |
| `src/submodules/vendor/gvhmr/body_model/data/smplx2smpl_sparse.pt` | `a95e94f6604dbfb112f01102a4ae939fa158eafc` | 3 | 5 | retain; alternate version |
| `src/submodules/vendor/gvhmr/body_model/data/smplx2smpl_sparse.pt` | `df81861451bbf90e45d22b4e0cdbee3aa70b581f` | 3820 | 208935 | remove |
| `src/submodules/vendor/gvhmr/body_model/data/smplx_verts437.pt` | `a95e94f6604dbfb112f01102a4ae939fa158eafc` | 3 | 5 | retain; alternate version |
| `src/submodules/vendor/gvhmr/body_model/data/smplx_verts437.pt` | `dbf87a143a4bcee0fbfb0600ba4d2d8190ce7422` | 3820 | 4203 | remove |
| `src/submodules/vendor/gvhmr/hmr2/smpl_mean_params.npz` | `a95e94f6604dbfb112f01102a4ae939fa158eafc` | 3 | 5 | retain; alternate version |
| `src/submodules/vendor/gvhmr/hmr2/smpl_mean_params.npz` | `b59913714b1cef7728994b6898348762b2179dae` | 3820 | 1310 | remove |

別版は共通の5-byte blob `a95e94f6604dbfb112f01102a4ae939fa158eafc`（内容は `asset`）で、各パス3 commit。承認された追加除去対象ではないため保持する。承認blobの除去後にこの仮データを過去の別commitへ復活させないことも、全tree比較で検証した。
