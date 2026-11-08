"""Get the PR's *head* code onto disk so it can be indexed.

Retrieval must see the code as the PR leaves it: chunk line numbers have to line up with
the diff's. In GitHub Actions, `actions/checkout` already provides that; locally this module
downloads the commit's archive.

The archive is UNTRUSTED input (anyone can open a PR against a public repo). It is only ever
parsed, never executed, and extraction is strict:
  * `tarfile.extract` is never used: each member is read and written by hand,
  * only regular `.py` files are written (no symlinks, devices, or other types),
  * paths with `..`, absolute paths, or anything resolving outside the destination are skipped,
  * excluded directories (virtualenvs, node_modules...) are skipped,
  * limits on archive size, file size and file count bound disk and time.
"""

import logging
import tarfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from pr_agent.retriever import EXCLUDED_DIRS

logger = logging.getLogger(__name__)

MAX_ARCHIVE_BYTES = 200 * 1024 * 1024
MAX_FILE_BYTES = 1_000_000
MAX_FILES = 20_000
_CHUNK = 1024 * 1024


class SnapshotError(RuntimeError):
    """The snapshot could not be fetched or exceeded a safety limit."""


def extract_python_sources(
    archive: Path, destination: Path, *,
    max_file_bytes: int = MAX_FILE_BYTES, max_files: int = MAX_FILES,
) -> int:
    """Write the archive's `.py` files under `destination`; return how many."""
    destination = destination.resolve()
    written = 0
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            if not member.isreg():  # directories, symlinks, hardlinks, devices: never written
                continue
            full = PurePosixPath(member.name).parts
            if member.name.startswith("/") or ".." in full:  # checked BEFORE dropping the top folder
                continue
            parts = full[1:]  # drop the archive's single top folder
            if not parts:
                continue
            if not parts[-1].endswith(".py") or any(part in EXCLUDED_DIRS for part in parts[:-1]):
                continue
            if member.size > max_file_bytes:
                logger.info("Skipping %s: %d bytes is over the file limit", "/".join(parts), member.size)
                continue
            target = destination.joinpath(*parts).resolve()
            if not target.is_relative_to(destination):  # belt and braces after the checks above
                continue
            if written >= max_files:
                logger.warning("Snapshot stopped at the %d-file limit", max_files)
                break
            source = tar.extractfile(member)
            if source is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("wb") as out:
                out.write(source.read(max_file_bytes + 1)[:max_file_bytes])
            written += 1
    return written


def download_archive(url: str, destination: Path, get: Callable[..., object] | None = None,
                     max_bytes: int = MAX_ARCHIVE_BYTES) -> None:
    """Stream `url` to `destination`, refusing anything over `max_bytes`."""
    if get is None:
        import requests

        get = requests.get
    response = get(url, stream=True, timeout=60)  # type: ignore[operator]
    response.raise_for_status()  # type: ignore[attr-defined]
    total = 0
    with destination.open("wb") as out:
        for block in response.iter_content(_CHUNK):  # type: ignore[attr-defined]
            total += len(block)
            if total > max_bytes:
                raise SnapshotError(f"archive is larger than the {max_bytes // (1024 * 1024)} MB limit")
            _ = out.write(block)


def fetch_snapshot(repository: object, sha: str, workdir: Path) -> Path:
    """Download `repository` at `sha` into a new directory under `workdir` and return it.

    `workdir` should be a fresh temporary directory; the scripts that use the result are
    kept elsewhere, so nothing in the snapshot can shadow an import.
    """
    url = repository.get_archive_link("tarball", ref=sha)  # type: ignore[attr-defined]
    archive = workdir / "snapshot.tar.gz"
    download_archive(url, archive)
    root = workdir / "repo"
    root.mkdir()
    count = extract_python_sources(archive, root)
    archive.unlink(missing_ok=True)
    logger.info("Snapshot of %s at %s: %d Python files", getattr(repository, "full_name", "?"), sha[:8], count)
    return root
