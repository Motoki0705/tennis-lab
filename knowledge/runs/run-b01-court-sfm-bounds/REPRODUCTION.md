The queue repro files retain the historical checkout and environment snapshot.
The original snapshot included a local NHT runtime symlink and omitted new,
untracked files. For exact rendering replay, use the persisted public camera
requests (`v3-request.json`, `sfm_bounded-request.json`) with `nht-render`, the
unchanged B01 scene export, and `CUDA_VISIBLE_DEVICES=0`, through the shared
training queue. Choose a fresh output directory. The gzipped plans preserve
all geometry and jitter metadata. The saved scripts document selection and
scoring; no pipeline alignment acceptance is asserted by this experiment.
