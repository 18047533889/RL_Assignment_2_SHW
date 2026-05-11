"""Tests for checkpoint path resolution (local relative -> absolute, remote pass-through)."""
import os

from rllib_fs_path import resolve_checkpoint_path, resolve_rllib_filesystem_path


def test_relative_dir_becomes_absolute():
    p = resolve_checkpoint_path("checkpoints")
    assert os.path.isabs(p)
    assert os.path.basename(os.path.normpath(p)) == "checkpoints"


def test_expanduser_tilde():
    p = resolve_checkpoint_path("~/tmp_ckpt_test")
    assert os.path.isabs(p)
    assert "tmp_ckpt_test" in p


def test_remote_uri_unchanged():
    assert resolve_checkpoint_path("s3://bucket/prefix/ckpt") == "s3://bucket/prefix/ckpt"
    assert resolve_checkpoint_path("gs://x/y") == "gs://x/y"


def test_file_uri_unchanged():
    u = "file:///tmp/foo/bar"
    assert resolve_checkpoint_path(u) == u


def test_legacy_alias_works():
    p = resolve_rllib_filesystem_path("checkpoints")
    assert os.path.isabs(p)
