"""Content-bound canonical checkpoint inventory used by native loading."""

import hashlib
import json
import os
from pathlib import Path
import stat

SUFFIXES = {".json", ".safetensors", ".model", ".txt", ".yaml", ".yml", ".jinja"}


def protected_path(path):
    """Require storage that the serving user and other users cannot rewrite."""
    path = Path(path).absolute()
    resolved = path.resolve(strict=True)
    for entry in {path, *path.parents, resolved, *resolved.parents}:
        info = entry.lstat()
        if stat.S_ISLNK(info.st_mode):
            continue
        readonly_mount = bool(os.statvfs(entry).f_flag & os.ST_RDONLY)
        if (info.st_mode & 0o022 or os.access(entry, os.W_OK, effective_ids=True)
                or (info.st_uid == os.geteuid() and not readonly_mount)):
            raise ValueError(
                "Checkpoint paths must be protected from serving-user and shared writes"
            )


def checkpoint_inventory(path):
    requested = Path(path)
    if not requested.is_absolute():
        raise ValueError("Canonical checkpoint path must be absolute")
    absolute = requested.absolute()
    root = requested.resolve(strict=True)
    if absolute != root:
        raise ValueError("Canonical checkpoint path and ancestors must not be symbolic links")
    if not root.is_dir():
        raise ValueError("Canonical checkpoint must be a local directory")
    files = {}
    for entry in root.iterdir():
        if entry.is_symlink():
            raise ValueError("Canonical checkpoint files must not be symbolic links")
        if entry.is_file() and entry.suffix in SUFFIXES:
            files[entry.name] = entry
    if "config.json" not in files or not any(name.endswith(".safetensors") for name in files):
        raise ValueError("Canonical checkpoint needs config.json and safetensors weights")
    index = files.get("model.safetensors.index.json")
    if index is not None:
        data = json.loads(index.read_text())
        shards = set(data.get("weight_map", {}).values())
        if not shards or not shards <= files.keys() or not all(
            name.endswith(".safetensors") for name in shards
        ):
            raise ValueError("Checkpoint shard index is incomplete or refers outside the directory")
    return root, files


def _file_identity(path):
    info = path.stat()
    return (
        info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
        info.st_ctime_ns, info.st_mode, info.st_uid, info.st_gid,
    )


def _stat_inventory(files):
    return {name: _file_identity(path) for name, path in sorted(files.items())}


def _digest(path, identity):
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        before_id = (
            before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
            before.st_ctime_ns, before.st_mode, before.st_uid, before.st_gid,
        )
        if before_id != identity or not stat.S_ISREG(before.st_mode):
            raise ValueError("Checkpoint changed before content verification")
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = os.fstat(stream.fileno())
        after_id = (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
            after.st_ctime_ns, after.st_mode, after.st_uid, after.st_gid,
        )
        if after_id != identity:
            raise ValueError("Checkpoint changed during content verification")
    return digest


class CheckpointSnapshot:
    """Hash a stable complete inventory, then cheaply guard later native reads."""

    def __init__(self, path, require_protected=True):
        self.root, files = checkpoint_inventory(path)
        if require_protected:
            protected_path(self.root)
            for entry in files.values():
                protected_path(entry)
        before = _stat_inventory(files)
        digests = {name: _digest(files[name], identity)
                   for name, identity in before.items()}
        root, after_files = checkpoint_inventory(self.root)
        after = _stat_inventory(after_files)
        if root != self.root or before != after:
            raise ValueError("Checkpoint changed while its content snapshot was created")
        self.identity = before
        self.digests = digests
        self.protected = require_protected
        self.content_identity = hashlib.sha256(
            json.dumps(digests, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        source_bytes = sum(
            identity[2] for name, identity in before.items()
            if name.endswith(".safetensors")
        )
        if not source_bytes:
            raise ValueError("Canonical checkpoint has no safetensors weight bytes")

    def check(self):
        root, files = checkpoint_inventory(self.root)
        if root != self.root or _stat_inventory(files) != self.identity:
            raise ValueError("Canonical checkpoint changed since content verification")

    def check_content(self):
        self.check()
        _, files = checkpoint_inventory(self.root)
        actual = {
            name: _digest(files[name], identity)
            for name, identity in self.identity.items()
        }
        self.check()
        if actual != self.digests:
            raise ValueError("Canonical checkpoint content differs from its startup snapshot")
