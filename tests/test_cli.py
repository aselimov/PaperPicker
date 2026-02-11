from pathlib import Path

from paper_picker.cli import get_default_config_path, parse_args


def test_default_config_path_falls_back_to_home_config(monkeypatch):
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert get_default_config_path() == Path.home() / ".config" / "paper_picker.toml"


def test_default_config_path_uses_xdg_config_home(monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", "~/my-config")
    assert get_default_config_path() == Path.home() / "my-config" / "paper_picker.toml"


def test_parse_args_default_config(monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", "~/alt-config")
    monkeypatch.setattr("sys.argv", ["paper-picker", "-n", "3"])
    args = parse_args()
    assert args.config == Path.home() / "alt-config" / "paper_picker.toml"
