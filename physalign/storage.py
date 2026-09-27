"""Strict JSON, content fingerprints, confined paths and durable local files."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def loads(text: str):
    return json.loads(text, object_pairs_hook=_object, parse_constant=_constant)


def read_json(path: Path):
    return loads(path.read_text(encoding="utf-8-sig"))


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint(value) -> str:
    return digest(canonical(value).encode("utf-8"))


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def confined(root: Path, relative: str) -> Path:
    """Reject absolute paths, drive paths, traversal and symlink escapes."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Bundle paths must be nonempty POSIX relative paths")
    p = PurePosixPath(relative)
    if p.is_absolute() or PureWindowsPath(relative).drive or any(x in {".", "..", ""} for x in relative.split("/")):
        raise ValueError(f"Unsafe relative path: {relative}")
    resolved = (root.resolve() / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes bundle: {relative}")
    return resolved


def outside_bundle(output: Path, dataset: Path) -> None:
    if output.resolve().is_relative_to(dataset.resolve()):
        raise ValueError("Plans and run outputs must be outside the unchanged source bundle")


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value) -> None:
    atomic_text(path, canonical(value) + "\n")


@contextmanager
def run_lock(directory: Path):
    """OS lock is released on process death; a stale filename is harmless."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("This run is already locked by another process") from exc
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("This run is already locked by another process") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
