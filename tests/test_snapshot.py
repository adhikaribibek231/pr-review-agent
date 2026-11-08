import io
import tarfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from pr_agent.snapshot import SnapshotError, download_archive, extract_python_sources, fetch_snapshot


def make_archive(path: Path, members: list[tuple[str, bytes | None, str]]) -> Path:
    """members: (name, content, kind) where kind is file | symlink | dir."""
    with tarfile.open(path, "w:gz") as tar:
        for name, content, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "symlink":
                info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
                tar.addfile(info)
            elif kind == "dir":
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            else:
                data = content or b""
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return path


def extract(tmp_path: Path, members: list[tuple[str, bytes | None, str]], **limits: int) -> tuple[int, Path]:
    archive = make_archive(tmp_path / "a.tar.gz", members)
    dest = tmp_path / "out"
    dest.mkdir()
    return extract_python_sources(archive, dest, **limits), dest


def test_python_files_are_extracted_without_the_archives_top_folder(tmp_path: Path) -> None:
    count, dest = extract(tmp_path, [("owner-repo-abc/", None, "dir"),
                                      ("owner-repo-abc/pkg/mod.py", b"x = 1\n", "file")])
    assert count == 1 and (dest / "pkg" / "mod.py").read_text() == "x = 1\n"


def test_other_file_types_are_not_written(tmp_path: Path) -> None:
    _, dest = extract(tmp_path, [("r/readme.md", b"# hi", "file"), ("r/data.json", b"{}", "file"),
                                  ("r/a.py", b"1", "file")])
    assert [p.name for p in dest.rglob("*") if p.is_file()] == ["a.py"]


@pytest.mark.parametrize("name", ["r/../../escape.py", "../escape.py", "/abs/escape.py", "r/a/../../../escape.py"])
def test_paths_that_try_to_leave_the_destination_are_skipped(tmp_path: Path, name: str) -> None:
    count, dest = extract(tmp_path, [(name, b"evil = 1", "file"), ("r/ok.py", b"ok = 1", "file")])
    assert count == 1
    assert not (tmp_path / "escape.py").exists() and not (tmp_path.parent / "escape.py").exists()
    assert [p.name for p in dest.rglob("*.py")] == ["ok.py"]


def test_symlinks_are_never_written_even_if_named_like_python_files(tmp_path: Path) -> None:
    count, dest = extract(tmp_path, [("r/link.py", None, "symlink"), ("r/ok.py", b"1", "file")])
    assert count == 1 and not (dest / "link.py").exists()


def test_virtualenvs_and_vendored_directories_are_skipped(tmp_path: Path) -> None:
    count, _ = extract(tmp_path, [("r/.venv/lib/x.py", b"1", "file"), ("r/node_modules/y.py", b"1", "file"),
                                   ("r/src/z.py", b"1", "file")])
    assert count == 1


def test_oversized_files_are_skipped(tmp_path: Path) -> None:
    count, dest = extract(tmp_path, [("r/big.py", b"x" * 5000, "file"), ("r/small.py", b"x", "file")],
                          max_file_bytes=1000)
    assert count == 1 and not (dest / "big.py").exists()


def test_the_file_count_is_capped(tmp_path: Path) -> None:
    count, _ = extract(tmp_path, [(f"r/f{i}.py", b"1", "file") for i in range(10)], max_files=3)
    assert count == 3


def test_an_empty_archive_yields_no_files(tmp_path: Path) -> None:
    assert extract(tmp_path, [])[0] == 0


# ---- download ---------------------------------------------------------------------------------------------


def response(blocks: list[bytes]) -> Mock:
    r = Mock()
    r.iter_content.return_value = blocks
    return r


def test_the_download_is_streamed_to_disk(tmp_path: Path) -> None:
    get = Mock(return_value=response([b"abc", b"def"]))
    download_archive("https://x/y", tmp_path / "a", get=get)
    assert (tmp_path / "a").read_bytes() == b"abcdef"
    assert get.call_args.kwargs["stream"] is True and get.call_args.kwargs["timeout"] == 60


def test_an_archive_over_the_limit_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SnapshotError, match="larger than"):
        download_archive("https://x/y", tmp_path / "a", get=Mock(return_value=response([b"x" * 600, b"x" * 600])),
                         max_bytes=1000)


def test_http_errors_propagate(tmp_path: Path) -> None:
    bad = response([])
    bad.raise_for_status.side_effect = RuntimeError("404")
    with pytest.raises(RuntimeError):
        download_archive("https://x/y", tmp_path / "a", get=Mock(return_value=bad))


def test_fetch_snapshot_downloads_extracts_and_cleans_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = make_archive(tmp_path / "src.tgz", [("o-r-1/main.py", b"print(1)\n", "file")])
    repo = Mock(full_name="o/r")
    repo.get_archive_link.return_value = "https://codeload/x"
    monkeypatch.setattr("pr_agent.snapshot.download_archive",
                        lambda url, dest, **_: dest.write_bytes(archive.read_bytes()))
    work = tmp_path / "work"
    work.mkdir()

    root = fetch_snapshot(repo, "abc123", work)

    repo.get_archive_link.assert_called_once_with("tarball", ref="abc123")
    assert (root / "main.py").exists() and not (work / "snapshot.tar.gz").exists()
