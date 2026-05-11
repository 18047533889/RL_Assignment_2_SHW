"""Resolve local checkpoint paths to absolute; pass through remote URIs (s3://, gs://)."""

from __future__ import annotations

import os


def resolve_checkpoint_path(path: str) -> str:
    p = path.strip()
    if "://" in p:
        return p
    return os.path.abspath(os.path.expanduser(p))


resolve_rllib_filesystem_path = resolve_checkpoint_path
