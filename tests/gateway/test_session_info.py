"""Tests for GatewayRunner._format_session_info — session config surfacing."""

import json

import pytest
from unittest.mock import patch

from gateway.run import GatewayRunner
from gateway.run_turn import _router_resolved_for_tier, _router_resolved_model_for_tier


@pytest.fixture()
def runner():
    """Create a bare GatewayRunner without __init__."""
    return GatewayRunner.__new__(GatewayRunner)


def _patch_info(tmp_path, config_yaml, model, runtime):
    """Return a context-manager stack that patches _format_session_info deps."""
    cfg_path = tmp_path / "config.yaml"
    if config_yaml is not None:
        cfg_path.write_text(config_yaml)
    return (
        patch("gateway.run._hermes_home", tmp_path),
        patch("gateway.run._resolve_gateway_model", return_value=model),
        patch("gateway.run._resolve_runtime_agent_kwargs", return_value=runtime),
    )


class TestFormatSessionInfo:



    def test_config_context_length(self, runner, tmp_path):
        p1, p2, p3 = _patch_info(tmp_path, "model:\n  default: test-model\n  context_length: 32768\n",
                                  "test-model",
                                  {"provider": "custom", "base_url": "", "api_key": ""})
        with p1, p2, p3:
            info = runner._format_session_info()
        assert "32K" in info
        assert "config" in info


    def test_local_endpoint_shown(self, runner, tmp_path):
        p1, p2, p3 = _patch_info(
            tmp_path,
            "model:\n  default: qwen3:8b\n  provider: custom\n  base_url: http://localhost:11434/v1\n  context_length: 8192\n",
            "qwen3:8b",
            {"provider": "custom", "base_url": "http://localhost:11434/v1", "api_key": ""})
        with p1, p2, p3:
            info = runner._format_session_info()
        assert "localhost:11434" in info
        assert "8K" in info

    def test_moa_preset_names_the_billed_aggregator(self, runner, tmp_path):
        """#112359: the preset name hides who pays; /model must name the acting aggregator."""
        p1, p2, p3 = _patch_info(tmp_path, "model:\n  default: review\n  provider: moa\n",
                                  "review", {"provider": "moa", "base_url": "", "api_key": ""})
        moa_cfg = {"moa": {"presets": {"review": {
            "reference_models": [{"provider": "openai", "model": "gpt-5.5"}],
            "aggregator": {"provider": "nous", "model": "claude-opus-4.8"},
        }}}}
        with p1, p2, p3, patch("hermes_cli.config.load_config", return_value=moa_cfg):
            info = runner._format_session_info()
        assert "nous:claude-opus-4.8" in info

    def test_named_custom_provider_keeps_context_pin_without_model_base_url(
        self, runner, tmp_path
    ):
        """Session-reset banner must honor model.context_length for named custom providers.

        Repro: /status shows 262144 from config while the reset banner said
        ``131K tokens (detected)`` because empty model.base_url + runtime URL
        falsely cleared the pin and fell through to the Qwen family default.
        """
        model = "custom-local-agentw/Qwen-AgentWorld-35B-A3B-Q5_K_XL"
        config_yaml = (
            "model:\n"
            f"  default: {model}\n"
            "  provider: custom-local-agentw\n"
            "  context_length: 262144\n"
            "custom_providers:\n"
            "  - name: custom-local-agentw\n"
            "    base_url: http://127.0.0.1:8080/v1\n"
            "    models: {}\n"
        )
        p1, p2, p3 = _patch_info(
            tmp_path,
            config_yaml,
            model,
            {
                "provider": "custom-local-agentw",
                "base_url": "http://127.0.0.1:8080/v1",
                "api_key": "",
            },
        )
        with p1, p2, p3, patch(
            "hermes_cli.config.get_compatible_custom_providers",
            return_value=[
                {
                    "name": "custom-local-agentw",
                    "base_url": "http://127.0.0.1:8080/v1",
                    "models": {},
                }
            ],
        ), patch(
            "agent.model_metadata.get_model_context_length",
            side_effect=lambda *args, **kwargs: (
                kwargs.get("config_context_length")
                if kwargs.get("config_context_length")
                else 131072
            ),
        ):
            info = runner._format_session_info()
        assert "262K" in info
        assert "config" in info
        assert "131K" not in info


class TestRouterTierModelBanner:
    """The banner must name the concrete model behind a shienga router tier."""

    def _snapshot(self, tmp_path, model="hermes-medium", resolved="deepseek-v4.1-flash", **entry):
        payload = {"provider": "ocg-coder-5", "model": resolved, "executor": "opencode-tfny", "executor_label": "OPENCODE-TFNY"}
        payload.update(entry)
        path = tmp_path / "resolved.json"
        path.write_text(json.dumps({"resolved": {model: payload}}))
        return path

    def test_helper_reads_the_tier_resolution(self, tmp_path):
        snapshot = self._snapshot(tmp_path)
        with patch("gateway.run_turn._router_resolved_path", return_value=snapshot):
            assert _router_resolved_model_for_tier("hermes-medium") == "deepseek-v4.1-flash"

    def test_helper_reads_the_serving_account(self, tmp_path):
        snapshot = self._snapshot(tmp_path)
        with patch("gateway.run_turn._router_resolved_path", return_value=snapshot):
            assert _router_resolved_for_tier("hermes-medium")["provider"] == "OPENCODE-TFNY"

    def test_helper_provider_falls_back_to_executor_then_family(self, tmp_path):
        no_label = self._snapshot(tmp_path, executor_label=None)
        with patch("gateway.run_turn._router_resolved_path", return_value=no_label):
            assert _router_resolved_for_tier("hermes-medium")["provider"] == "opencode-tfny"

    def test_helper_ignores_non_tier_models(self, tmp_path):
        snapshot = self._snapshot(tmp_path, model="hermes-medium")
        with patch("gateway.run_turn._router_resolved_path", return_value=snapshot):
            assert _router_resolved_model_for_tier("deepseek-v4.1-flash") == ""
            assert _router_resolved_for_tier("deepseek-v4.1-flash") == {}

    def test_helper_is_empty_when_snapshot_missing(self, tmp_path):
        with patch("gateway.run_turn._router_resolved_path", return_value=tmp_path / "missing.json"):
            assert _router_resolved_model_for_tier("hermes-medium") == ""
            assert _router_resolved_for_tier("hermes-medium") == {}

    def test_banner_appends_the_routed_model_after_the_tier(self, runner, tmp_path):
        snapshot = self._snapshot(tmp_path)
        p1, p2, p3 = _patch_info(
            tmp_path,
            "model:\n  default: hermes-medium\n  provider: custom\n  base_url: http://127.0.0.1:19876/v1\n",
            "hermes-medium",
            {"provider": "custom", "base_url": "http://127.0.0.1:19876/v1", "api_key": ""},
        )
        with p1, p2, p3, patch("gateway.run_turn._router_resolved_path", return_value=snapshot):
            info = runner._format_session_info()
        assert "hermes-medium" in info
        assert "deepseek-v4.1-flash" in info
        # The router hop must not be reported as the provider.
        assert "Provider: custom" not in info
        assert "OPENCODE-TFNY" in info

    def test_banner_unchanged_for_a_concrete_model(self, runner, tmp_path):
        snapshot = self._snapshot(tmp_path)
        p1, p2, p3 = _patch_info(
            tmp_path,
            "model:\n  default: deepseek-v4.1-flash\n  provider: custom\n  base_url: http://127.0.0.1:19876/v1\n",
            "deepseek-v4.1-flash",
            {"provider": "custom", "base_url": "http://127.0.0.1:19876/v1", "api_key": ""},
        )
        with p1, p2, p3, patch("gateway.run_turn._router_resolved_path", return_value=snapshot):
            info = runner._format_session_info()
        assert "deepseek-v4.1-flash" in info
        # No second, routed model appended for an already-concrete model.
        assert info.count("deepseek-v4.1-flash") == 1


class TestResetNoticeSessionInfo:
    """#59003: the auto-reset banner must report the serving profile's config,
    not the multiplexer's base config."""

    _RUNTIME = {"provider": "", "base_url": "", "api_key": ""}

    def _source(self):
        from gateway.config import Platform
        from gateway.session import SessionSource
        return SessionSource(
            platform=Platform.TELEGRAM, chat_id="123", user_id="u1",
            profile="planner",
        )

    def _homes(self, tmp_path):
        base = tmp_path / "base"
        profile = tmp_path / "profiles" / "planner"
        profile.mkdir(parents=True)
        base.mkdir()
        base.joinpath("config.yaml").write_text(
            "model:\n  default: base-model\n  provider: custom\n  context_length: 1000\n")
        profile.joinpath("config.yaml").write_text(
            "model:\n  default: profile-model\n  provider: anthropic\n  context_length: 2000\n")
        return base, profile

    def test_multiplex_uses_profile_config(self, runner, tmp_path):
        from types import SimpleNamespace
        base, profile = self._homes(tmp_path)
        runner.config = SimpleNamespace(multiplex_profiles=True)
        with patch("gateway.run._hermes_home", base), \
             patch.object(GatewayRunner, "_resolve_profile_home_for_source", return_value=profile), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value=self._RUNTIME):
            info = runner._reset_notice_session_info(self._source())
        assert "profile-model" in info
        assert "anthropic" in info
        assert "base-model" not in info

