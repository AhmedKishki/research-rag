"""MOBI admission, read-only decoding, locators and fail-closed parser errors."""

import struct
import sys
import time

import pytest

from research_rag.corpus.extraction import ExtractionError, extract_sources
from research_rag.corpus.mobi_reader import MobiReadError, read_mobi
from research_rag.corpus.sources import SourcePolicyError, scan_sources, sha256_file
from research_rag.project.config import resolve_config
from tests.mobi_fixtures import (
    dual_mobi_bytes,
    huff_mobi_bytes,
    install_small_pipe_worker,
    kf8_bytes,
    kf8_fragment_bytes,
    mobi_bytes,
    write_mobi,
)


@pytest.mark.parametrize("compression", [1, 2])
def test_mobi_extracts_metadata_html_and_locators_read_only(project, compression):
    path = project / "sources" / "book.MOBI"
    write_mobi(path, compression=compression)
    digest = sha256_file(path)
    config = resolve_config(project, vanilla_executable=sys.executable)
    source = scan_sources(config).selected[0]
    assert source.extension == ".mobi"
    documents, units = extract_sources((source,))
    record = documents[0]
    assert record["format"] == "mobi"
    assert record["title"] == "Synthetic Evidence"
    assert record["authors"] == ["Example Author"]
    assert record["year"] == 2024
    assert record["language"] == ["en"]
    assert record["metadata_provenance"]["title"] == "mobi_header"
    assert record["mobi_compression"] == compression
    assert len(units) == 2
    assert "café" in units[0]["contents"]
    assert "author’s" in units[0]["contents"]
    assert all(unit["locator"]["type"] == "mobi_section" for unit in units)
    assert units[0]["locator"]["fragment"] == "opening"
    assert units[1]["locator"]["section_title"] == "Second argument"
    assert units[1]["locator"]["block_index"] == 2
    assert "SCRIPT" not in " ".join(unit["contents"] for unit in units)
    assert sha256_file(path) == digest
    assert sorted(p.name for p in path.parent.iterdir()) == ["book.MOBI"]


@pytest.mark.parametrize(
    "damage",
    [
        "drm",
        "short",
        "offset",
        "length",
        "encoding",
        "compression",
        "exth",
        "dictionary",
        "replica",
        "kf8",
    ],
)
def test_mobi_refuses_drm_corruption_and_unsupported_variants(project, damage):
    data = bytearray(mobi_bytes())
    header = struct.unpack_from(">I", data, 78)[0]
    if damage == "drm":
        struct.pack_into(">H", data, header + 12, 2)
    elif damage == "short":
        data = data[:60]
    elif damage == "offset":
        struct.pack_into(">I", data, 86, 1)
    elif damage == "length":
        struct.pack_into(">I", data, header + 4, 12345)
    elif damage == "encoding":
        struct.pack_into(">I", data, header + 0x1C, 999)
    elif damage == "compression":
        struct.pack_into(">H", data, header, 999)
    elif damage == "exth":
        struct.pack_into(">I", data, header + 0xF4 + 16, 1)
    elif damage == "dictionary":
        struct.pack_into(">I", data, header + 0x28, 1)
    elif damage == "replica":
        text = struct.unpack_from(">I", data, 86)[0]
        data[text : text + 4] = b"%MOP"
    elif damage == "kf8":
        struct.pack_into(">I", data, header + 0x24, 8)
    path = project / "sources" / "bad.mobi"
    path.write_bytes(data)
    config = resolve_config(project, vanilla_executable=sys.executable)
    with pytest.raises(ExtractionError, match="MOBI"):
        extract_sources(scan_sources(config).selected)


def test_mobi_does_not_admit_source_symlinks(project, tmp_path):
    target = tmp_path / "outside.mobi"
    write_mobi(target)
    (project / "sources" / "linked.mobi").symlink_to(target)
    config = resolve_config(project, vanilla_executable=sys.executable)
    with pytest.raises(SourcePolicyError, match="Symbolic-link"):
        scan_sources(config)
    with pytest.raises(MobiReadError, match="non-symlink"):
        read_mobi(project / "sources" / "linked.mobi")


def test_kf8_reconstructs_real_skeleton_index_without_output_files(project):
    path = project / "sources" / "modern.mobi"
    path.write_bytes(kf8_bytes())
    parsed = read_mobi(path)
    assert parsed.version == 8
    assert len(parsed.parts) == 1
    assert "cobalt institutions" in parsed.parts[0]
    config = resolve_config(project, vanilla_executable=sys.executable)
    documents, units = extract_sources(scan_sources(config).selected)
    assert documents[0]["mobi_version"] == 8
    assert len(units) == 2
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize(
    "builder,version,compression",
    [(huff_mobi_bytes, 6, 0x4448), (dual_mobi_bytes, 6, 1)],
)
def test_huff_and_dual_containers_decode_one_rendition(
    project, builder, version, compression
):
    path = project / "sources" / "book.mobi"
    path.write_bytes(builder())
    parsed = read_mobi(path)
    assert parsed.version == version
    assert parsed.compression == compression
    assert len(parsed.parts) == 1
    assert "cobalt institutions" in parsed.parts[0]


def test_dual_mobi_refuses_encrypted_second_rendition(project):
    path = project / "sources" / "protected.mobi"
    path.write_bytes(dual_mobi_bytes(second_drm=2))
    with pytest.raises(MobiReadError, match="DRM/encrypted"):
        read_mobi(path)


def test_mobi_timeout_is_explicit_and_uses_sanitized_environment(monkeypatch, tmp_path):
    from research_rag.corpus import mobi_reader

    original = mobi_reader.subprocess.Popen
    processes = []

    def observe(command, **kwargs):
        environment = mobi_reader.child_process_environment()
        environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        assert kwargs["env"] == environment
        process = original(command, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(mobi_reader.subprocess, "Popen", observe)
    monkeypatch.setattr(mobi_reader, "MOBI_TIMEOUT_SECONDS", 0)
    with pytest.raises(MobiReadError, match="45-second"):
        read_mobi(tmp_path / "book.mobi")
    assert processes[0].poll() is not None


def test_dual_mobi_refuses_unsupported_unused_compression(project):
    path = project / "sources" / "broken.mobi"
    path.write_bytes(dual_mobi_bytes(second_compression=999))
    with pytest.raises(MobiReadError, match="Unsupported MOBI compression"):
        read_mobi(path)


def test_tag_dense_mobi_never_reaches_parent_html_parser(project, monkeypatch):
    from research_rag.corpus import extraction

    path = project / "sources" / "dense.mobi"
    write_mobi(path, "<p>x</p>" * 100_000)

    def forbidden(*args, **kwargs):
        raise AssertionError("MOBI HTML reached the serving parent")

    monkeypatch.setattr(extraction, "BeautifulSoup", forbidden)
    config = resolve_config(project, vanilla_executable=sys.executable)
    with pytest.raises(ExtractionError, match="50,000-tag safety limit"):
        extract_sources(scan_sources(config).selected)


def test_ordinary_mobi_html_is_parsed_only_in_bounded_child(project, monkeypatch):
    from research_rag.corpus import extraction

    write_mobi(project / "sources" / "ordinary.mobi")

    def forbidden(*args, **kwargs):
        raise AssertionError("MOBI HTML reached the serving parent")

    monkeypatch.setattr(extraction, "BeautifulSoup", forbidden)
    config = resolve_config(project, vanilla_executable=sys.executable)
    _documents, units = extract_sources(scan_sources(config).selected)
    assert len(units) == 2


def test_mobi_ipc_is_bounded_before_parent_json_load(project, monkeypatch):
    from research_rag.corpus import mobi_reader

    path = project / "sources" / "ordinary.mobi"
    write_mobi(path)
    monkeypatch.setattr(mobi_reader, "MAX_MOBI_IPC_BYTES", 128)

    def forbidden(*args, **kwargs):
        raise AssertionError("Over-budget JSON reached the serving parent")

    monkeypatch.setattr(mobi_reader.json, "loads", forbidden)
    with pytest.raises(MobiReadError, match="IPC safety limit"):
        read_mobi(path)


def test_mobi_declared_size_limit_and_nonregular_files_fail_closed(tmp_path):
    data = bytearray(mobi_bytes())
    header = struct.unpack_from(">I", data, 78)[0]
    struct.pack_into(">I", data, header + 4, 64 * 1024 * 1024 + 1)
    path = tmp_path / "oversized.mobi"
    path.write_bytes(data)
    with pytest.raises(MobiReadError, match="64 MiB"):
        read_mobi(path)
    directory = tmp_path / "directory.mobi"
    directory.mkdir()
    with pytest.raises(MobiReadError, match="regular"):
        read_mobi(directory)


def test_huff_recursive_phrase_decodes_and_cycle_fails_closed(project):
    path = project / "sources" / "recursive.mobi"
    path.write_bytes(huff_mobi_bytes(recursive=True))
    assert "Research evidence" in read_mobi(path).parts[0]
    path.write_bytes(huff_mobi_bytes(cycle=True))
    with pytest.raises(MobiReadError, match="Cannot parse MOBI"):
        read_mobi(path)


def test_kf8_nonzero_fragment_reconstruction_and_corrupt_bounds(project):
    path = project / "sources" / "fragment.mobi"
    path.write_bytes(kf8_fragment_bytes())
    parsed = read_mobi(path)
    assert "cobalt institutions" in parsed.parts[0]
    assert '<h2 id="second">Second argument</h2>' in parsed.parts[0]
    config = resolve_config(project, vanilla_executable=sys.executable)
    _documents, units = extract_sources(scan_sources(config).selected)
    assert len(units) == 2
    path.write_bytes(kf8_fragment_bytes(corrupt_length=True))
    with pytest.raises(MobiReadError, match="Corrupt KF8 fragment bounds"):
        read_mobi(path)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="Linux pipe-size control"
)
def test_small_pipe_request_write_obeys_deadline_and_reaps_child(tmp_path, monkeypatch):
    from research_rag.corpus import mobi_reader

    processes = install_small_pipe_worker(
        monkeypatch,
        "import sys,time; time.sleep(1); sys.stdin.buffer.read(); print('{}')",
    )
    monkeypatch.setattr(mobi_reader, "MOBI_TIMEOUT_SECONDS", 0.05)
    started = time.monotonic()
    with pytest.raises(MobiReadError, match="45-second safety limit"):
        mobi_reader._worker_payload(
            tmp_path / "delayed.mobi", {"padding": "x" * 60_000}
        )
    # A blocking write would take at least the child's full one-second delay.
    assert time.monotonic() - started < 0.5
    assert len(processes) == 1
    assert processes[0].poll() is not None


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="Linux pipe-size control"
)
def test_early_worker_refusal_during_request_write_is_typed_and_preserves_diagnostics(
    tmp_path, monkeypatch
):
    from research_rag.corpus import mobi_reader

    processes = install_small_pipe_worker(
        monkeypatch,
        "import os,sys; os.close(0); sys.stderr.write('refused: corrupt source\\n'); sys.exit(1)",
    )
    with pytest.raises(MobiReadError, match="refused: corrupt source"):
        mobi_reader._worker_payload(
            tmp_path / "refused.mobi", {"padding": "x" * 60_000}
        )
    assert len(processes) == 1
    assert processes[0].poll() == 1
