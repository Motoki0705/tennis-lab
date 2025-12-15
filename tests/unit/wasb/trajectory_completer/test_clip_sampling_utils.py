from src.wasb.scripts import clip_sampling


def test_collect_kept_indices(tmp_path):
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    (samples_dir / "Clip_1.mp4").write_bytes(b"")
    (samples_dir / "Clip_10.mp4").write_bytes(b"")
    (samples_dir / "not_a_clip.mp4").write_bytes(b"")

    assert clip_sampling._collect_kept_indices(samples_dir) == {1, 10}


def test_reindex_clip_dirs(tmp_path):
    game_dir = tmp_path / "game11"
    game_dir.mkdir()

    (game_dir / "Clip1").mkdir()
    (game_dir / "Clip3").mkdir()
    (game_dir / "Clip10").mkdir()

    clip_sampling._reindex_clip_dirs(game_dir)

    assert (game_dir / "Clip1").exists()
    assert (game_dir / "Clip2").exists()
    assert (game_dir / "Clip3").exists()
    assert not (game_dir / "Clip10").exists()

