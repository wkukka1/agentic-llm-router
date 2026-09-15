from __future__ import annotations

from router.provenance import artifact_digests, file_digest, git_dirty, git_sha


def test_git_sha_and_dirty_here():
    # this repo is a git checkout, so both should resolve to concrete values
    sha = git_sha()
    assert isinstance(sha, str) and len(sha) == 40
    assert isinstance(git_dirty(), bool)


def test_file_digest_stable_and_length(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes(b"hello world")
    d1 = file_digest(p)
    d2 = file_digest(p)
    assert d1 == d2 and len(d1) == 16
    assert file_digest(p, n=8) == d1[:8]
    assert file_digest(p, algo="sha1") != d1          # different algo -> different digest
    assert file_digest(tmp_path / "missing") is None


def test_artifact_digests_maps_labels(tmp_path):
    (tmp_path / "x").write_bytes(b"x")
    out = artifact_digests({"x": tmp_path / "x", "gone": tmp_path / "gone"})
    assert set(out) == {"x", "gone"}
    assert out["x"] and out["gone"] is None
