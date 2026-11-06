"""SWAH-653: the loopback router must not warn as if it needed a credential.

The auxiliary resolver emits, per call, a WARNING that a named custom provider "has no resolvable
api_key — request will be sent with placeholder no-key-required and will 401 on auth-required
endpoints". That premise is false for the local Hermes router (127.0.0.1:19876): it authenticates
no caller, so the placeholder IS the correct credential. The warning was read as proof the router
needed a key while SWAH-651 was investigated, and it is identical to the string a genuinely broken
remote provider produces — so it must be narrowed, not deleted.
"""

from agent import auxiliary_client


def test_loopback_base_urls_are_recognised():
    for base in (
        "http://127.0.0.1:19876/v1",
        "http://127.0.0.1/v1",
        "http://127.5.5.5:8080/v1",
        "http://localhost:19876/v1",
        "http://LOCALHOST/v1",
        "http://[::1]:19876/v1",
        "http://0.0.0.0:19876/v1",
    ):
        assert auxiliary_client._base_url_is_loopback(base) is True, base


def test_remote_and_empty_base_urls_are_not_loopback():
    for base in (
        "https://router.example.com/v1",
        "https://ollama.com/v1",
        "http://10.0.0.5:19876/v1",
        "http://127.0.0.1.evil.example/v1",
        "",
        None,
    ):
        assert auxiliary_client._base_url_is_loopback(base) is False, base


def test_warning_is_suppressed_for_a_loopback_router():
    assert auxiliary_client._should_warn_missing_custom_key(
        "no-key-required", "http://127.0.0.1:19876/v1") is False


def test_warning_is_kept_for_a_remote_provider_without_a_key():
    assert auxiliary_client._should_warn_missing_custom_key(
        "no-key-required", "https://router.example.com/v1") is True


def test_loopback_with_a_real_credential_never_warns():
    assert auxiliary_client._should_warn_missing_custom_key(
        "synthetic-key-not-a-real-credential", "http://127.0.0.1:19876/v1") is False
