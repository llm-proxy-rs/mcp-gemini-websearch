"""Tests for Cognito OAuth configuration via OAuthProxy + JWTVerifier."""

import os
from unittest.mock import patch, MagicMock

import pytest
from fastmcp.server.auth import OAuthProxy
from fastmcp.server.auth.providers.jwt import JWTVerifier, RSAKeyPair
from key_value.aio.wrappers.encryption import FernetEncryptionWrapper

import server

# --- Helpers ---

_COGNITO_FULL_ENV = {
    "COGNITO_USER_POOL_ID": "us-east-1_Test",
    "COGNITO_REGION": "us-east-1",
    "COGNITO_CLIENT_ID": "client-id",
    "COGNITO_CLIENT_SECRET": "secret",
    "COGNITO_DOMAIN": "test",
}


def _build_auth(**env_overrides):
    """Call _build_cognito_auth with specific COGNITO env vars."""
    env = {**_COGNITO_FULL_ENV, **env_overrides}
    with patch.dict(os.environ, env, clear=False), patch.object(
        server, "MCP_BASE_URL", "https://example.com"
    ):
        return server._build_cognito_auth()


def _build_auth_missing(key: str):
    """Call _build_cognito_auth with one COGNITO var removed."""
    env = {k: v for k, v in _COGNITO_FULL_ENV.items() if k != key}
    # Clear all COGNITO_ vars first, then set only what we want
    clear = {k: "" for k in _COGNITO_FULL_ENV}
    with patch.dict(os.environ, {**clear, **env}, clear=False), patch.object(
        server, "MCP_BASE_URL", "https://example.com"
    ):
        return server._build_cognito_auth()


# --- Configuration tests ---


def test_no_auth_without_env_var():
    """When COGNITO_USER_POOL_ID is not set, _build_cognito_auth returns None."""
    clear = {k: "" for k in _COGNITO_FULL_ENV}
    with patch.dict(os.environ, clear, clear=False):
        result = server._build_cognito_auth()
    assert result is None


def test_auth_requires_region():
    """Missing COGNITO_REGION should raise RuntimeError."""
    with pytest.raises(RuntimeError, match="COGNITO_REGION"):
        _build_auth_missing("COGNITO_REGION")


def test_auth_requires_client_id():
    """Missing COGNITO_CLIENT_ID should raise RuntimeError."""
    with pytest.raises(RuntimeError, match="COGNITO_CLIENT_ID"):
        _build_auth_missing("COGNITO_CLIENT_ID")


def test_auth_requires_client_secret():
    """Missing COGNITO_CLIENT_SECRET should raise RuntimeError."""
    with pytest.raises(RuntimeError, match="COGNITO_CLIENT_SECRET"):
        _build_auth_missing("COGNITO_CLIENT_SECRET")


def test_auth_requires_domain():
    """Missing COGNITO_DOMAIN should raise RuntimeError."""
    with pytest.raises(RuntimeError, match="COGNITO_DOMAIN"):
        _build_auth_missing("COGNITO_DOMAIN")


def test_auth_configured_with_all_vars():
    """With all required vars set, should return an OAuthProxy instance."""
    result = _build_auth()
    assert isinstance(result, OAuthProxy)


# --- _build_storage tests ---


def test_build_storage_returns_none_without_database_url():
    """When DATABASE_URL is not set, _build_storage returns None."""
    with patch.dict(os.environ, {"DATABASE_URL": ""}, clear=False):
        result = server._build_storage("some-signing-key")
    assert result is None


def test_build_storage_raises_without_signing_key():
    """When DATABASE_URL is set but jwt_signing_key is empty, should raise."""
    with patch.dict(
        os.environ, {"DATABASE_URL": "postgresql://localhost/test"}, clear=False
    ):
        with pytest.raises(RuntimeError, match="MCP_JWT_SIGNING_KEY"):
            server._build_storage("")


@patch("server.PostgreSQLStore")
def test_build_storage_returns_fernet_wrapper(mock_pg_store):
    """When DATABASE_URL and signing key are set, should return FernetEncryptionWrapper."""
    mock_pg_store.return_value = MagicMock()
    with patch.dict(
        os.environ, {"DATABASE_URL": "postgresql://localhost/test"}, clear=False
    ):
        result = server._build_storage("a-valid-signing-key-for-testing-1234")
    assert isinstance(result, FernetEncryptionWrapper)
    mock_pg_store.assert_called_once()


@patch("server.PostgreSQLStore")
def test_build_storage_passes_database_url_to_pg_store(mock_pg_store):
    """PostgreSQLStore should be created with the DATABASE_URL."""
    mock_pg_store.return_value = MagicMock()
    db_url = "postgresql://user:pass@db.example.com:5432/mydb"
    with patch.dict(os.environ, {"DATABASE_URL": db_url}, clear=False):
        server._build_storage("a-valid-signing-key-for-testing-1234")
    mock_pg_store.assert_called_once_with(url=db_url)


# --- JWTVerifier token validation tests ---

ISSUER = "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_Test"
AUDIENCE = "test-client-id"


@pytest.fixture
def rsa_keys():
    return RSAKeyPair.generate()


@pytest.fixture
def verifier(rsa_keys):
    return JWTVerifier(
        public_key=rsa_keys.public_key,
        issuer=ISSUER,
        audience=AUDIENCE,
    )


@pytest.mark.asyncio
async def test_valid_token_accepted(verifier, rsa_keys):
    """A properly signed, unexpired token should be accepted."""
    token = rsa_keys.create_token(issuer=ISSUER, audience=AUDIENCE, scopes=["openid"])
    result = await verifier.verify_token(token)
    assert result is not None
    assert result.token == token
    assert "openid" in result.scopes


@pytest.mark.asyncio
async def test_expired_token_rejected(verifier, rsa_keys):
    """An expired token should be rejected."""
    token = rsa_keys.create_token(
        issuer=ISSUER,
        audience=AUDIENCE,
        expires_in_seconds=-60,
    )
    result = await verifier.verify_token(token)
    assert result is None


@pytest.mark.asyncio
async def test_wrong_issuer_rejected(verifier, rsa_keys):
    """A token with wrong issuer should be rejected."""
    token = rsa_keys.create_token(
        issuer="https://wrong-issuer.example.com", audience=AUDIENCE
    )
    result = await verifier.verify_token(token)
    assert result is None


@pytest.mark.asyncio
async def test_wrong_audience_rejected(verifier, rsa_keys):
    """A token with wrong audience should be rejected."""
    token = rsa_keys.create_token(issuer=ISSUER, audience="wrong-client")
    result = await verifier.verify_token(token)
    assert result is None


@pytest.mark.asyncio
async def test_wrong_key_rejected(rsa_keys):
    """A token signed with a different key should be rejected."""
    other_keys = RSAKeyPair.generate()
    verifier = JWTVerifier(
        public_key=rsa_keys.public_key,
        issuer=ISSUER,
        audience=AUDIENCE,
    )
    token = other_keys.create_token(issuer=ISSUER, audience=AUDIENCE)
    result = await verifier.verify_token(token)
    assert result is None


@pytest.mark.asyncio
async def test_garbage_token_rejected(verifier):
    """A completely invalid token string should be rejected."""
    result = await verifier.verify_token("not.a.valid.jwt")
    assert result is None


@pytest.mark.asyncio
async def test_required_scopes_enforced(rsa_keys):
    """A token missing required scopes should be rejected."""
    verifier = JWTVerifier(
        public_key=rsa_keys.public_key,
        issuer=ISSUER,
        audience=AUDIENCE,
        required_scopes=["admin"],
    )
    token = rsa_keys.create_token(issuer=ISSUER, audience=AUDIENCE, scopes=["openid"])
    result = await verifier.verify_token(token)
    assert result is None


@pytest.mark.asyncio
async def test_required_scopes_satisfied(rsa_keys):
    """A token with all required scopes should be accepted."""
    verifier = JWTVerifier(
        public_key=rsa_keys.public_key,
        issuer=ISSUER,
        audience=AUDIENCE,
        required_scopes=["openid", "profile"],
    )
    token = rsa_keys.create_token(
        issuer=ISSUER, audience=AUDIENCE, scopes=["openid", "profile", "email"]
    )
    result = await verifier.verify_token(token)
    assert result is not None
