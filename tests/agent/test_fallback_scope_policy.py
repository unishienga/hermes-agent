"""Which axis a fallback failure invalidates: credential, endpoint, or deployment.

``should_skip_candidate`` owns the axes themselves and is tested in
``test_backend_identity.py``. This file locks the CALLER's choice of axis: the
reason a route just failed decides whether a chain entry is skipped because it
shares a credential surface, an endpoint, or one model deployment.

The failure this pins: with every reason treated as model-scoped, a 401/402/429
on one account left the sibling models on the same credential surface in the
chain, so failover walked into the same dead credential. The carve-out in the
other direction matters just as much — a model-scoped failure (500, timeout,
upstream 429, missing model) must never condemn the sibling deployment.
"""

from unittest.mock import patch

import pytest

from agent.chat_completion_helpers import _should_skip_fallback_candidate
from agent.error_classifier import FailoverReason


class _Agent:
    """The three attributes the predicate reads off the live agent."""

    def __init__(self, provider="custom", model="zai-org/glm-5.2", base_url=""):
        self.provider = provider
        self.model = model
        self.base_url = base_url


# Same provider label, different model: one credential surface, two deployments.
# Only a credential-scoped failure may skip this entry.
SIBLING = {"provider": "custom", "model": "mindai/macaron-v1-venti", "base_url": ""}


def _skip(agent, fb, reason):
    """Call the predicate with the entitlement/cooldown/local-usability helpers neutralised.

    Those three read live config and credential state; this file is about the
    identity axis only, so they are held out of the way rather than faked.
    ``_is_entitlement_rejected`` is imported inside the function body, so it is
    patched at its owner (``agent.fallback_cooldown``), not at this module.
    """
    with (
        patch("agent.fallback_cooldown._is_entitlement_rejected", return_value=False),
        patch("agent.chat_completion_helpers._candidate_pool_exhausted", return_value=False),
        patch(
            "agent.chat_completion_helpers._fallback_entry_unavailable_without_network",
            return_value="",
        ),
    ):
        return _should_skip_fallback_candidate(
            agent,
            fb,
            (fb["provider"], fb["model"]),
            fb["provider"],
            fb["model"],
            set(),
            reason,
        )


@pytest.mark.parametrize(
    "reason",
    [
        FailoverReason.auth,
        FailoverReason.auth_permanent,
        FailoverReason.billing,
        FailoverReason.rate_limit,
    ],
)
def test_credential_failures_skip_the_sibling_on_the_same_credential_surface(reason):
    """One account's 401/402/429 is evidence against the credential, not the model."""
    assert _skip(_Agent(), SIBLING, reason) is True


@pytest.mark.parametrize(
    "reason",
    [
        FailoverReason.server_error,
        FailoverReason.timeout,
        FailoverReason.upstream_rate_limit,
        FailoverReason.model_not_found,
        FailoverReason.overloaded,
    ],
)
def test_model_scoped_failures_leave_the_sibling_deployment_reachable(reason):
    """A 500/timeout/upstream-429 says nothing about the sibling model."""
    assert _skip(_Agent(), SIBLING, reason) is False


def test_tls_failure_is_endpoint_scoped():
    """A TLS chain failure condemns the host, not the credential or the model."""
    agent = _Agent(provider="custom", model="m1", base_url="https://h:8443/v1")
    fb = {"provider": "other", "model": "m2", "base_url": "https://h:8443/v1"}
    assert _skip(agent, fb, FailoverReason.ssl_cert_verification) is True
    # The same entry survives a model-scoped failure: different deployment.
    assert _skip(agent, fb, FailoverReason.server_error) is False


def test_unknown_reason_is_model_scoped():
    """No reason means no evidence beyond the one deployment — never over-skip."""
    assert _skip(_Agent(), SIBLING, None) is False
