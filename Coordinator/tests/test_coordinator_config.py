"""Coordinator config.json: full option coverage, path resolution, env overlay."""

from pathlib import Path

from dain_coordinator.config import (
    find_base_dir,
    load_config,
    resolve_settings,
    write_default_config,
)
from dain_coordinator.settings import (
    COORDINATOR_ENV,
    DEFAULT_ADMIN_API_KEY,
    DEFAULT_API_KEY,
    DEFAULT_CONFIG,
    DEFAULT_JOIN_TOKEN,
    CoordinatorSettings,
    harden_production_secrets,
)


def test_round_trip_default_config() -> None:
    """Every DEFAULT_CONFIG key is read back into a settings field."""
    settings = CoordinatorSettings.from_config(DEFAULT_CONFIG, Path("."))
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.join_token == "Jj3L7ewD"
    assert settings.discovery_enabled is True
    assert settings.discovery_port == 8456
    assert settings.cors_origins == ("*",)
    assert settings.temp_degrade_c == 90.0
    assert settings.monitor_tick_s == 0.5
    assert settings.uptime_alpha == 0.1
    assert settings.overload_strikes_to_degrade == 3


def test_full_override() -> None:
    data = {
        **DEFAULT_CONFIG,
        "host": "192.168.1.50",
        "port": 9000,
        "heartbeat_interval_s": 1.0,
        "offline_after_missed": 5,
        "join_token": "custom",
        "min_score": 0.2,
        "log_json": "true",
        "queue_limit": 3,
        "max_completion_tokens": 64,
        "cors_origins": ["http://a", "http://b"],
        "rate_limit_per_min": 0,
        "admin_api_key": "admin2",
        "discovery_enabled": "false",
        "unrelated_option": "ignored",  # unknown keys are ignored
    }
    settings = CoordinatorSettings.from_config(data, Path("."))
    assert settings.host == "192.168.1.50"
    assert settings.port == 9000
    assert settings.heartbeat_interval_s == 1.0
    assert settings.offline_after_missed == 5
    assert settings.join_token == "custom"
    assert settings.min_score == 0.2
    assert settings.log_json is True
    assert settings.queue_limit == 3
    assert settings.max_completion_tokens == 64
    assert settings.cors_origins == ("http://a", "http://b")
    assert settings.rate_limit_per_min == 0
    assert settings.admin_api_key == "admin2"
    assert settings.discovery_enabled is False


def test_cors_accepts_comma_string() -> None:
    settings = CoordinatorSettings.from_config(
        {"cors_origins": "http://a, http://b"}, Path(".")
    )
    assert settings.cors_origins == ("http://a", "http://b")


def test_relative_paths_resolve_against_base_dir() -> None:
    base = Path("E:/coord")
    settings = CoordinatorSettings.from_config(
        {"db_path": "data/coord.sqlite3", "model_store_dir": "models"}, base
    )
    assert settings.db_path == str(base / "data" / "coord.sqlite3")
    assert settings.model_store_dir == str(base / "models")


def test_absolute_paths_kept() -> None:
    settings = CoordinatorSettings.from_config(
        {"db_path": "C:/data/coord.sqlite3"}, Path("E:/coord")
    )
    assert Path(settings.db_path) == Path("C:/data/coord.sqlite3")


def test_write_default_config_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    write_default_config(path)
    assert load_config(path) == DEFAULT_CONFIG


def test_missing_config_file_returns_empty() -> None:
    assert load_config(Path("does-not-exist.json")) == {}


def test_resolve_creates_default_when_missing(tmp_path: Path) -> None:
    settings, path, created = resolve_settings(tmp_path)
    assert created is True
    assert path == tmp_path / "config.json"
    assert path.is_file()
    assert settings.port == 8000
    assert settings.db_path == str(tmp_path / "coordinator.sqlite3")


def test_resolve_uses_existing_config(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        '{"port": 9123, "join_token": "file-token"}', encoding="utf-8"
    )
    settings, _, created = resolve_settings(tmp_path)
    assert created is False
    assert settings.port == 9123
    assert settings.join_token == "file-token"


def test_resolve_env_overrides_config(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "config.json").write_text('{"port": 9123}', encoding="utf-8")
    monkeypatch.setenv("DAIN_PORT", "7777")
    settings, _, _ = resolve_settings(tmp_path)
    assert settings.port == 7777


def test_resolve_env_overrides_missing_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DAIN_JOIN_TOKEN", "env-token")
    settings, _, created = resolve_settings(tmp_path)
    assert created is True
    assert settings.join_token == "env-token"


def test_env_map_covers_every_default_key() -> None:
    missing = set(DEFAULT_CONFIG) - set(COORDINATOR_ENV)
    assert not missing, f"no env override mapping for: {sorted(missing)}"


def test_find_base_dir_points_to_project_root() -> None:
    assert (find_base_dir() / "dain_coordinator" / "__init__.py").is_file()


def test_harden_replaces_default_secrets_when_exposed() -> None:
    settings = harden_production_secrets(CoordinatorSettings(host="0.0.0.0"))
    assert settings.join_token != DEFAULT_JOIN_TOKEN
    assert settings.api_key != DEFAULT_API_KEY
    assert settings.admin_api_key != DEFAULT_ADMIN_API_KEY
    # Generated values must actually be credentials, not empty.
    assert len(settings.api_key) >= 16


def test_harden_keeps_defaults_on_loopback() -> None:
    settings = harden_production_secrets(CoordinatorSettings(host="127.0.0.1"))
    assert settings.join_token == DEFAULT_JOIN_TOKEN
    assert settings.api_key == DEFAULT_API_KEY
    assert settings.admin_api_key == DEFAULT_ADMIN_API_KEY


def test_harden_keeps_explicit_secrets() -> None:
    settings = harden_production_secrets(
        CoordinatorSettings(
            host="0.0.0.0",
            join_token="explicit-join",
            api_key="explicit-api",
            admin_api_key="explicit-admin",
        )
    )
    assert settings.join_token == "explicit-join"
    assert settings.api_key == "explicit-api"
    assert settings.admin_api_key == "explicit-admin"


def test_harden_replaces_empty_secrets() -> None:
    settings = harden_production_secrets(
        CoordinatorSettings(host="0.0.0.0", join_token="", api_key="", admin_api_key="")
    )
    assert settings.join_token != ""
    assert settings.api_key != ""
    assert settings.admin_api_key != ""


def test_resolve_hardens_default_secrets(tmp_path: Path) -> None:
    # No config file and no env → the exposed 0.0.0.0 bind must not keep the
    # source-checked-in defaults.
    settings, _, _ = resolve_settings(tmp_path)
    assert settings.join_token != DEFAULT_JOIN_TOKEN
    assert settings.api_key != DEFAULT_API_KEY
    assert settings.admin_api_key != DEFAULT_ADMIN_API_KEY
