"""The worker's lifetime stays bounded even when its parent is interrupted."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from research_rag.corpus import docling_worker, pdf_backend
from research_rag.corpus.docling_env import DoclingEnv

_FAKE_ENV = DoclingEnv(
    root=Path("/nonexistent/docling-env"),
    spec_fingerprint="spec-test",
    versions={"docling": "2.135.0"},
)


@pytest.mark.parametrize("interrupted", [False, True])
def test_worker_scope_is_retired_on_success_or_interruption(monkeypatch, interrupted):
    commands = []
    retired = []

    class Process:
        returncode = 0

        def communicate(self, *args, **kwargs):
            if interrupted:
                raise KeyboardInterrupt
            return b"{}", b""

    def launch(command, **kwargs):
        commands.append(command)
        return Process()

    monkeypatch.setattr(pdf_backend, "_require_systemd_run", lambda: "/bin/systemd-run")
    monkeypatch.setattr(pdf_backend.subprocess, "Popen", launch)
    monkeypatch.setattr(
        pdf_backend, "_terminate_scope", lambda unit, process: retired.append(unit)
    )
    arguments = {
        "source_path": Path("/tmp/source.pdf"),
        "start": 1,
        "end": 1,
        "offline": True,
        "model_cache_root": None,
        "docling_env": _FAKE_ENV,
        "timeout": 23,
    }
    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            pdf_backend.run_docling_batch(**arguments)
    else:
        assert pdf_backend.run_docling_batch(**arguments) == {}

    assert len(retired) == 1
    assert "--property=RuntimeMaxSec=23" in commands[0]


def test_worker_requires_scope_cleanup_executable(monkeypatch):
    monkeypatch.setattr(
        pdf_backend.shutil,
        "which",
        lambda name: "/bin/systemd-run" if name == "systemd-run" else None,
    )
    with pytest.raises(pdf_backend.DoclingUnavailableError, match="systemctl"):
        pdf_backend._require_systemd_run()


def test_converter_explicitly_disables_enrichments_and_gpu(monkeypatch):
    captured = {}

    def converter(**kwargs):
        captured.update(kwargs)
        return captured

    modules = {
        "docling.datamodel.base_models": {"InputFormat": SimpleNamespace(PDF="pdf")},
        "docling.datamodel.pipeline_options": {
            "PdfPipelineOptions": lambda: SimpleNamespace(
                accelerator_options=SimpleNamespace()
            )
        },
        "docling.document_converter": {
            "DocumentConverter": converter,
            "PdfFormatOption": lambda **kwargs: SimpleNamespace(**kwargs),
        },
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    docling_worker._build_converter()
    options = captured["format_options"]["pdf"].pipeline_options
    for flag in (
        "do_ocr",
        "do_table_structure",
        "do_code_enrichment",
        "do_formula_enrichment",
        "do_picture_classification",
        "do_picture_description",
        "do_chart_extraction",
        "generate_page_images",
        "generate_picture_images",
        "generate_table_images",
        "enable_remote_services",
    ):
        assert getattr(options, flag) is False
    assert options.accelerator_options.device == "cpu"
    assert options.accelerator_options.num_threads == 2
