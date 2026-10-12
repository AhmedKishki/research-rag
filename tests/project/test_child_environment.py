"""Purpose-specific child environments reject ambient secrets and preserve essentials."""

import os

import pytest

from research_rag.project.config import child_process_environment

PURPOSES = [
    "parser",
    "docling-offline",
    "docling",
    "provisioner",
    "viewer",
    "systemctl",
]


@pytest.mark.parametrize("purpose", PURPOSES)
def test_children_reject_credentials_and_execution_overrides(monkeypatch, purpose):
    rejected = [
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SESSION_TOKEN",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "OPENAI_API_KEY",
        "GITHUB_TOKEN",
        "DATABASE_URL",
        "ARBITRARY_SECRET",
        "SSH_AUTH_SOCK",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "UV_INDEX_URL",
        "PIP_INDEX_URL",
        "RESEARCH_RAG_UI_PORT",
        "LC_SECRET",
    ]
    for name in rejected:
        monkeypatch.setenv(name, "must-not-leak")
    monkeypatch.setenv("RESEARCH_RAG_OWNER_PID", "999999")
    environment = child_process_environment(purpose=purpose)
    assert not set(rejected).intersection(environment)
    assert environment["RESEARCH_RAG_MANAGED_CHILD"] == "1"
    assert environment["RESEARCH_RAG_OWNER_PID"] == str(os.getpid())


@pytest.mark.parametrize("purpose", PURPOSES)
def test_children_preserve_required_variables(monkeypatch, purpose):
    required = {
        "PATH": "/usr/bin",
        "HOME": "/home/reader",
        "USER": "reader",
        "LOGNAME": "reader",
        "LANG": "en_US.UTF-8",
        "LANGUAGE": "en",
        "LC_ALL": "C.UTF-8",
        "LC_MESSAGES": "C",
        "TZ": "UTC",
        "TMPDIR": "/tmp/private",
        "TMP": "/tmp/private",
        "TEMP": "/tmp/private",
        "XDG_CACHE_HOME": "/cache",
        "XDG_CONFIG_HOME": "/config",
        "XDG_DATA_HOME": "/data",
        "XDG_STATE_HOME": "/state",
    }
    session = {
        "XDG_RUNTIME_DIR": "/run/user/1000",
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/bus",
    }
    desktop = {
        "DISPLAY": ":0",
        "WAYLAND_DISPLAY": "wayland-0",
        "XAUTHORITY": "/auth",
        "XDG_CURRENT_DESKTOP": "GNOME",
        "XDG_DATA_DIRS": "/usr/share",
    }
    network = {
        "HTTPS_PROXY": "http://proxy:8080",
        "https_proxy": "http://proxy:8080",
        "SSL_CERT_FILE": "/cert.pem",
        "REQUESTS_CA_BUNDLE": "/ca.pem",
    }
    monkeypatch.setattr(os, "environ", required | session | desktop | network)
    environment = child_process_environment(purpose=purpose)
    assert required.items() <= environment.items()
    assert (
        (session.items() <= environment.items())
        if purpose != "parser"
        else not set(session).intersection(environment)
    )
    assert (
        (desktop.items() <= environment.items())
        if purpose == "viewer"
        else not set(desktop).intersection(environment)
    )
    assert (
        (network.items() <= environment.items())
        if purpose in {"docling", "provisioner"}
        else not set(network).intersection(environment)
    )


def test_unknown_purpose_is_rejected():
    with pytest.raises(ValueError, match="purpose"):
        child_process_environment(purpose="unknown")


def test_docling_and_installer_environments_reject_secrets(monkeypatch, tmp_path):
    from research_rag.corpus import docling_env, pdf_backend

    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:8080")
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/bus")
    offline = pdf_backend._worker_environment(pdf_backend.MODE_EXTRACT, True, tmp_path)
    online = pdf_backend._worker_environment(pdf_backend.MODE_EXTRACT, False, tmp_path)
    installer = docling_env._install_environment()
    for environment in (offline, online, installer):
        assert "HF_TOKEN" not in environment
        assert environment["DBUS_SESSION_BUS_ADDRESS"] == "unix:path=/bus"
    assert "HTTPS_PROXY" not in offline
    assert offline["HF_HUB_OFFLINE"] == "1"
    assert online["HTTPS_PROXY"] == installer["HTTPS_PROXY"] == "http://proxy:8080"


@pytest.mark.parametrize("module_name", ["docling_env", "pdf_backend"])
def test_scope_cleanup_uses_filtered_session_environment(monkeypatch, module_name):
    from types import SimpleNamespace

    from research_rag.corpus import docling_env, pdf_backend

    module = {"docling_env": docling_env, "pdf_backend": pdf_backend}[module_name]
    monkeypatch.setenv("GITHUB_TOKEN", "secret")
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
    monkeypatch.setattr(module.shutil, "which", lambda name: "/usr/bin/systemctl")
    observed = []
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: observed.append(kw))
    monkeypatch.setattr(module.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(module.os, "killpg", lambda *args: None)
    module._terminate_scope(
        "test.scope", SimpleNamespace(pid=123, wait=lambda **kw: None)
    )
    assert "GITHUB_TOKEN" not in observed[0]["env"]
    assert observed[0]["env"]["XDG_RUNTIME_DIR"] == "/run/user/1000"
