"""MCP server that provides web search via Gemini Grounding with Google Search."""

import json
import logging
import os

import httpx

from cryptography.fernet import Fernet
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route
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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

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
        log.warning(
            "No DATABASE_URL set — OAuth state will not persist across restarts"
        )
        return None
    if not jwt_signing_key:
        raise RuntimeError(
            "MCP_JWT_SIGNING_KEY must be set when DATABASE_URL is configured"
        )
    storage_encryption_salt = os.getenv("STORAGE_ENCRYPTION_SALT")
    if not storage_encryption_salt:
        raise RuntimeError(
            "STORAGE_ENCRYPTION_SALT must be set when DATABASE_URL is configured"
        )
    encryption_key = derive_jwt_key(
        high_entropy_material=jwt_signing_key,
        salt=storage_encryption_salt,
    )
    log.info("Using PostgreSQL for OAuth storage")
    global _oauth_storage
    _oauth_storage = FernetEncryptionWrapper(
        PostgreSQLStore(url=database_url),
        fernet=Fernet(key=encryption_key),
    )
    return _oauth_storage


def _require_env(name: str) -> str:
    value = os.getenv(name)
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

    scopes_raw = _require_env("COGNITO_SCOPES")
    required_scopes = [s.strip() for s in scopes_raw.split() if s.strip()]

    # Stable random string for signing JWTs. Generate with: openssl rand -base64 32
    jwt_signing_key = _require_env("MCP_JWT_SIGNING_KEY")
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
        client_storage=_build_storage(jwt_signing_key),
    )

    log.info("Cognito OAuth proxy enabled")
    return auth


_oauth_storage: FernetEncryptionWrapper | None = None
_cognito_auth = _build_cognito_auth()  # may set _oauth_storage via _build_storage

mcp = FastMCP("gemini-websearch", auth=_cognito_auth)


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


async def _sanitize_url(url: str) -> str:
    """Resolve Google's vertexaisearch redirect to the actual source URL.

    Only follows a single redirect from Google's own domain — does NOT
    connect to the destination. Returns the original URL on failure.
    """
    parsed = httpx.URL(url)
    if parsed.host != "vertexaisearch.cloud.google.com" or not parsed.path.startswith(
        "/grounding-api-redirect/"
    ):
        return str(parsed)
    try:
        async with httpx.AsyncClient(follow_redirects=False, timeout=5) as client:
            resp = await client.head(url)
        location = resp.headers.get("location")
        if not location:
            return ""
        resolved = httpx.URL(location)
        if resolved.scheme in ("http", "https") and resolved.host:
            return str(resolved)
        return ""
    except Exception:
        log.debug("Failed to resolve redirect for URL")
        return ""


async def _format_response(response, query: str) -> str:
    try:
        text = response.text or ""
    except Exception:
        text = ""

    if not getattr(response, "candidates", None):
        return text

    metadata = getattr(response.candidates[0], "grounding_metadata", None)
    if not metadata:
        return text

    chunks = getattr(metadata, "grounding_chunks", None) or []

    # Build links array from grounding chunks, stripping redirect URLs.
    links = []
    for chunk in chunks:
        web = getattr(chunk, "web", None)
        if web:
            uri = await _sanitize_url(getattr(web, "uri", ""))
            if uri:
                links.append(
                    {
                        "title": getattr(web, "title", ""),
                        "url": uri,
                    }
                )

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
    try:
        response = await _gemini.aio.models.generate_content(
            model=GEMINI_MODEL, contents=query, config=_GOOGLE_SEARCH_CONFIG
        )
        return await _format_response(response, query)
    except Exception:
        log.exception("web_search failed for query: %s", query)
        raise RuntimeError(
            "Web search is temporarily unavailable. Please try again later."
        )


# ── Entrypoint ───────────────────────────────────────────────────────────────


async def health(request):
    """Health check endpoint. Pings the database when configured."""
    if _oauth_storage is None:
        return PlainTextResponse("ok")
    try:
        pg_store = _oauth_storage.key_value  # underlying PostgreSQLStore
        await pg_store.setup()
        await pg_store._pool.fetchval("SELECT 1")
        return JSONResponse({"status": "ok", "db": "ok"})
    except Exception:
        log.exception("Health check: database unreachable")
        return JSONResponse(
            {"status": "degraded", "db": "unreachable"}, status_code=503
        )


if __name__ == "__main__":
    import uvicorn

    log.info("Starting Gemini Web Search MCP server (streamable-http)")

    app = mcp.http_app(transport="streamable-http", stateless_http=True)
    app.routes.append(Route("/health", health))

    uvicorn.run(app, host=MCP_HOST, port=MCP_PORT)
