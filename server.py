"""MCP server that provides web search via Gemini Grounding with Google Search."""

import json
import logging
import os

from cryptography.fernet import Fernet
from fastmcp import FastMCP
from fastmcp.server.auth import OAuthProxy
from fastmcp.server.auth.jwt_issuer import derive_jwt_key
from fastmcp.server.auth.providers.jwt import JWTVerifier
from google import genai
from google.genai import types
from key_value.aio.stores.postgresql import PostgreSQLStore
from key_value.aio.wrappers.encryption import FernetEncryptionWrapper

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "8080"))
MCP_BASE_URL = os.getenv("MCP_BASE_URL", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
if not GOOGLE_CLOUD_PROJECT:
    raise RuntimeError("GOOGLE_CLOUD_PROJECT must be set")

_gemini = genai.Client(
    vertexai=True,
    project=GOOGLE_CLOUD_PROJECT,
    location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
)


# ── Auth ─────────────────────────────────────────────────────────────────────


def _build_storage(jwt_signing_key: str) -> FernetEncryptionWrapper | None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return None
    if not jwt_signing_key:
        raise RuntimeError(
            "MCP_JWT_SIGNING_KEY must be set when DATABASE_URL is configured"
        )
    encryption_key = derive_jwt_key(
        high_entropy_material=jwt_signing_key,
        salt="fastmcp-storage-encryption-key",
    )
    return FernetEncryptionWrapper(
        store=PostgreSQLStore(url=database_url),
        fernet=Fernet(key=encryption_key),
    )


def _require_env(name: str) -> str:
    value = os.getenv(name, "")
    if not value:
        raise RuntimeError(
            f"{name} must be set when COGNITO_USER_POOL_ID is configured"
        )
    return value


def _build_cognito_auth() -> OAuthProxy | None:
    pool_id = os.getenv("COGNITO_USER_POOL_ID")
    if not pool_id:
        log.info("No COGNITO_USER_POOL_ID set — running without auth")
        return None

    region = _require_env("COGNITO_REGION")
    client_id = _require_env("COGNITO_CLIENT_ID")
    client_secret = _require_env("COGNITO_CLIENT_SECRET")
    domain = _require_env("COGNITO_DOMAIN")

    cognito_base_url = f"https://{domain}.auth.{region}.amazoncognito.com"
    issuer_url = f"https://cognito-idp.{region}.amazonaws.com/{pool_id}"

    scopes_raw = os.getenv("COGNITO_SCOPES", "")
    required_scopes = [s.strip() for s in scopes_raw.split() if s.strip()] or None

    # Stable random string for signing JWTs. Generate with: openssl rand -base64 32
    jwt_signing_key = os.getenv("MCP_JWT_SIGNING_KEY")
    auth = OAuthProxy(
        upstream_authorization_endpoint=f"{cognito_base_url}/oauth2/authorize",
        upstream_token_endpoint=f"{cognito_base_url}/oauth2/token",
        upstream_client_id=client_id,
        upstream_client_secret=client_secret,
        token_verifier=JWTVerifier(
            jwks_uri=f"{issuer_url}/.well-known/jwks.json",
            issuer=issuer_url,
            required_scopes=required_scopes,
        ),
        base_url=MCP_BASE_URL,
        token_endpoint_auth_method="client_secret_basic",
        forward_pkce=True,
        jwt_signing_key=jwt_signing_key,
        client_storage=_build_storage(jwt_signing_key or ""),
    )

    log.info(
        "Cognito OAuth proxy enabled: pool=%s region=%s domain=%s",
        pool_id,
        region,
        domain,
    )
    return auth


mcp = FastMCP("gemini-websearch", auth=_build_cognito_auth())


# ── Response formatting ──────────────────────────────────────────────────────
#
# Wrap Gemini's grounded response in the same envelope as Claude Code's
# WebSearch tool result:
#
#   Web search results for query: "python 3.14"
#
#   Links: [{"title":"Python.org","url":"https://python.org"}, ...]
#
#   Python 3.14 was released in October 2025. ...
#
#   REMINDER: You MUST include the sources above in your response ...


def _format_response(response, query: str) -> str:
    text = response.text or ""

    if not response.candidates:
        return text

    metadata = response.candidates[0].grounding_metadata
    if not metadata:
        return text

    chunks = metadata.grounding_chunks or []

    # Build links array from grounding chunks.
    links = [
        {"title": chunk.web.title, "url": chunk.web.uri}
        for chunk in chunks
        if chunk.web
    ]

    # Assemble the tool result in the same format as Claude Code's WebSearch.
    parts = [f'Web search results for query: "{query}"']
    if links:
        parts.append(f"Links: {json.dumps(links)}")
    else:
        parts.append("No links found.")
    parts.append(text)
    parts.append(
        "REMINDER: You MUST include the sources above in your response "
        "to the user using markdown hyperlinks."
    )

    return "\n\n".join(parts)


# ── Tools ────────────────────────────────────────────────────────────────────

_GOOGLE_SEARCH_CONFIG = types.GenerateContentConfig(
    tools=[types.Tool(google_search=types.GoogleSearch())],
)


@mcp.tool()
async def web_search(query: str) -> str:
    """Search the web using Gemini Grounding with Google Search.

    - Allows Claude to search the web and use the results to inform responses
    - Provides up-to-date information for current events and recent data
    - Returns search result information formatted as search result blocks, including links as markdown hyperlinks
    - Use this tool for accessing information beyond Claude's knowledge cutoff
    - Searches are performed automatically within a single API call

    CRITICAL REQUIREMENT - You MUST follow this:
      - After answering the user's question, you MUST include a "Sources:" section at the end of your response
      - In the Sources section, list all relevant URLs from the search results as markdown hyperlinks: [Title](URL)
      - This is MANDATORY - never skip including sources in your response
      - Example format:

        [Your answer here]

        Sources:
        - [Source Title 1](https://example.com/1)
        - [Source Title 2](https://example.com/2)

    Usage notes:
      - Domain filtering is supported to include or block specific websites
      - Web search is only available in the US

    IMPORTANT - Use the correct year in search queries:
      - Today's date is 2025-12-06. You MUST use this year when searching for recent information, documentation, or current events.
      - Example: If today is 2025-07-15 and the user asks for "latest React docs", search for "React documentation 2025", NOT "React documentation 2024"
    """
    response = await _gemini.aio.models.generate_content(
        model=GEMINI_MODEL, contents=query, config=_GOOGLE_SEARCH_CONFIG
    )
    return _format_response(response, query)


# ── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route
    import uvicorn

    log.info("Starting Gemini Web Search MCP server (streamable-http)")

    app = mcp.http_app(transport="streamable-http")
    app.routes.append(Route("/health", lambda _: PlainTextResponse("ok")))

    uvicorn.run(app, host=MCP_HOST, port=MCP_PORT)
