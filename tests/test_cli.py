from __future__ import annotations

import pytest

from quantum_ai_lab import __version__
from quantum_ai_lab.cli import DEMOS, main


def test_tour_runs_every_module(capsys):
    assert main(["--no-color", "tour"]) == 0
    out = capsys.readouterr().out
    assert "[8/8]" in out
    assert "Tour finished" in out


def test_list_shows_all_demos(capsys):
    assert main(["--no-color"]) == 0
    out = capsys.readouterr().out
    for name in DEMOS:
        assert name in out


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert __version__ in capsys.readouterr().out


def test_demo_modules_are_importable():
    import importlib

    for module_name, _ in DEMOS.values():
        assert callable(importlib.import_module(module_name).main)
