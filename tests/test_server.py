"""Tests for the Gemini WebSearch MCP server."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from starlette.routing import Route
from starlette.testclient import TestClient

from server import _format_response, _sanitize_url, health, mcp, web_search

# ---------------------------------------------------------------------------
# Helpers to build mock Gemini response objects
# ---------------------------------------------------------------------------


def _make_chunk(title: str, uri: str) -> SimpleNamespace:
    return SimpleNamespace(web=SimpleNamespace(title=title, uri=uri))


def _make_response(
    text: str,
    *,
    metadata=None,
) -> SimpleNamespace:
    """Build a minimal mock Gemini response."""
    candidate = SimpleNamespace(grounding_metadata=metadata)
    return SimpleNamespace(text=text, candidates=[candidate])


# Patch _sanitize_url to passthrough in tests (no real HTTP calls).
async def _passthrough_sanitize(url):
    return url


_noop_sanitize = patch("server._sanitize_url", side_effect=_passthrough_sanitize)


# ---------------------------------------------------------------------------
# 1. _format_response  (pure logic)
# ---------------------------------------------------------------------------


class TestFormatResponse:
    @pytest.mark.asyncio
    async def test_text_only_no_metadata(self):
        resp = _make_response("Hello world")
        assert await _format_response(resp, "test") == "Hello world"

    @pytest.mark.asyncio
    async def test_text_only_metadata_none(self):
        resp = _make_response("Hello", metadata=None)
        assert await _format_response(resp, "test") == "Hello"

    @pytest.mark.asyncio
    async def test_no_grounding_just_text(self):
        metadata = SimpleNamespace(
            web_search_queries=["q1", "q2"],
            grounding_chunks=None,
            grounding_supports=None,
        )
        resp = _make_response("answer", metadata=metadata)
        result = await _format_response(resp, "test")
        assert "answer" in result
        assert "No links found." in result
        assert "REMINDER:" in result

    @pytest.mark.asyncio
    @_noop_sanitize
    async def test_links_and_reminder_included(self, _mock):
        chunks = [
            _make_chunk("Site A", "https://a.com"),
            _make_chunk("Site B", "https://b.com"),
        ]
        metadata = SimpleNamespace(
            web_search_queries=None,
            grounding_chunks=chunks,
            grounding_supports=None,
        )
        resp = _make_response("info", metadata=metadata)
        result = await _format_response(resp, "my query")
        assert result.startswith('Web search results for query: "my query"')
        assert '"title": "Site A"' in result
        assert '"url": "https://a.com"' in result
        assert '"title": "Site B"' in result
        assert "info" in result
        assert "REMINDER:" in result

    @pytest.mark.asyncio
    async def test_empty_candidates(self):
        resp = SimpleNamespace(text="Hello", candidates=[])
        assert await _format_response(resp, "test") == "Hello"

    @pytest.mark.asyncio
    async def test_no_links_says_no_links(self):
        metadata = SimpleNamespace(
            web_search_queries=None,
            grounding_chunks=[],
            grounding_supports=None,
        )
        resp = _make_response("answer", metadata=metadata)
        result = await _format_response(resp, "test")
        assert "No links found." in result

    @pytest.mark.asyncio
    @_noop_sanitize
    async def test_query_appears_in_envelope(self, _mock):
        chunks = [_make_chunk("X", "https://x.com")]
        metadata = SimpleNamespace(
            web_search_queries=None,
            grounding_chunks=chunks,
            grounding_supports=None,
        )
        resp = _make_response("text", metadata=metadata)
        result = await _format_response(resp, "kubernetes gateway api")
        assert 'Web search results for query: "kubernetes gateway api"' in result


# ---------------------------------------------------------------------------
# 2. _sanitize_url
# ---------------------------------------------------------------------------


class TestSanitizeUrl:
    @pytest.mark.asyncio
    async def test_normal_url_passes_through(self):
        assert (
            await _sanitize_url("https://example.com/page")
            == "https://example.com/page"
        )

    @pytest.mark.asyncio
    async def test_vertexai_redirect_is_resolved(self):
        redirect_url = (
            "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc123"
        )

        mock_resp = SimpleNamespace(
            headers={"location": "https://www.example.com/article"}
        )
        with patch("server.httpx.AsyncClient") as mock_cls:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_ctx.head = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_ctx

            result = await _sanitize_url(redirect_url)
            assert result == "https://www.example.com/article"

    @pytest.mark.asyncio
    async def test_vertexai_redirect_blocks_non_http(self):
        redirect_url = (
            "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc123"
        )

        mock_resp = SimpleNamespace(headers={"location": "file:///etc/passwd"})
        with patch("server.httpx.AsyncClient") as mock_cls:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_ctx.head = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_ctx

            result = await _sanitize_url(redirect_url)
            assert result == ""

    @pytest.mark.asyncio
    async def test_vertexai_redirect_failure_returns_empty(self):
        redirect_url = (
            "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc123"
        )

        with patch("server.httpx.AsyncClient") as mock_cls:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_ctx.head = AsyncMock(side_effect=Exception("timeout"))
            mock_cls.return_value = mock_ctx

            result = await _sanitize_url(redirect_url)
            assert result == ""

    @pytest.mark.asyncio
    async def test_url_with_spaces_is_encoded(self):
        result = await _sanitize_url(
            "https://www.example.com/search?q=best+python tips+and tricks"
        )
        assert (
            result == "https://www.example.com/search?q=best+python%20tips+and%20tricks"
        )

    @pytest.mark.asyncio
    async def test_vertexai_non_redirect_path_passes_through(self):
        url = "https://vertexaisearch.cloud.google.com/other-path"
        result = await _sanitize_url(url)
        assert result == url

    @pytest.mark.asyncio
    async def test_vertexai_redirect_encodes_resolved_url(self):
        redirect_url = (
            "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc"
        )
        mock_resp = SimpleNamespace(
            headers={"location": "https://example.com/path with spaces"}
        )
        with patch("server.httpx.AsyncClient") as mock_cls:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_ctx.head = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_ctx

            result = await _sanitize_url(redirect_url)
            assert result == "https://example.com/path%20with%20spaces"

    @pytest.mark.asyncio
    async def test_vertexai_redirect_no_location_header(self):
        redirect_url = (
            "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc123"
        )

        mock_resp = SimpleNamespace(headers={})
        with patch("server.httpx.AsyncClient") as mock_cls:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_ctx.head = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_ctx

            result = await _sanitize_url(redirect_url)
            assert result == ""

    @pytest.mark.asyncio
    async def test_empty_url_passes_through(self):
        assert await _sanitize_url("") == ""


# ---------------------------------------------------------------------------
# 3. web_search tool
# ---------------------------------------------------------------------------


class TestWebSearch:
    def test_description_contains_todays_date(self):
        assert date.today().isoformat() in web_search.__doc__

    @pytest.mark.asyncio
    async def test_calls_gemini_and_returns_text(self):
        mock_response = _make_response("search result")

        with patch("server._gemini") as mock_client:
            mock_client.aio.models.generate_content = AsyncMock(
                return_value=mock_response
            )
            result = await web_search("test query")

        # No metadata → falls through to plain text
        assert result == "search result"

        call_kwargs = mock_client.aio.models.generate_content.call_args
        assert call_kwargs.kwargs["contents"] == "test query"

    @pytest.mark.asyncio
    async def test_api_error_returns_generic_message(self):
        with patch("server._gemini") as mock_client:
            mock_client.aio.models.generate_content = AsyncMock(
                side_effect=Exception(
                    "Failed to retrieve http://metadata.google.internal/... "
                    "service-account@project.iam.gserviceaccount.com"
                )
            )
            with pytest.raises(RuntimeError, match="temporarily unavailable"):
                await web_search("test query")

    @pytest.mark.asyncio
    @_noop_sanitize
    async def test_returns_envelope_with_links(self, _mock):
        chunks = [_make_chunk("Example", "https://example.com")]
        metadata = SimpleNamespace(
            web_search_queries=["test"],
            grounding_chunks=chunks,
            grounding_supports=None,
        )
        mock_response = _make_response("result text", metadata=metadata)

        with patch("server._gemini") as mock_client:
            mock_client.aio.models.generate_content = AsyncMock(
                return_value=mock_response
            )
            result = await web_search("test")

        assert 'Web search results for query: "test"' in result
        assert "result text" in result
        assert '"title": "Example"' in result
        assert "REMINDER:" in result


# ---------------------------------------------------------------------------
# 4. MCP server integration
# ---------------------------------------------------------------------------


class TestMCPIntegration:
    @pytest.mark.asyncio
    async def test_server_exposes_web_search_tool(self):
        tools = await mcp.list_tools()
        tool_names = {t.name for t in tools}
        assert "web_search" in tool_names

    @pytest.mark.asyncio
    async def test_call_web_search_through_mcp(self):
        mock_response = _make_response("mcp result")

        with patch("server._gemini") as mock_client:
            mock_client.aio.models.generate_content = AsyncMock(
                return_value=mock_response
            )
            result = await mcp.call_tool("web_search", {"query": "test"})

        assert result.content[0].text == "mcp result"


# ---------------------------------------------------------------------------
# 5. Health endpoint
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_ok_without_db(self):
        """When no DB is configured, health returns plain 'ok'."""
        with patch("server._oauth_storage", None):
            app = mcp.http_app(transport="streamable-http")
            app.routes.append(Route("/health", health))
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.text == "ok"

    def test_health_returns_ok_with_db(self):
        """When DB is configured and reachable, health returns JSON ok."""
        mock_pool = AsyncMock()
        mock_pool.fetchval = AsyncMock(return_value=1)

        mock_store = AsyncMock()
        mock_store._pool = mock_pool

        mock_storage = AsyncMock()
        mock_storage.key_value = mock_store

        with patch("server._oauth_storage", mock_storage):
            app = mcp.http_app(transport="streamable-http")
            app.routes.append(Route("/health", health))
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/health")

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "db": "ok"}
        mock_store.setup.assert_awaited_once()
        mock_pool.fetchval.assert_awaited_once_with("SELECT 1")

    def test_health_returns_503_when_db_unreachable(self):
        """When DB is configured but unreachable, health returns 503."""
        mock_store = AsyncMock()
        mock_store.setup.side_effect = ConnectionError("connection refused")

        mock_storage = AsyncMock()
        mock_storage.key_value = mock_store

        with patch("server._oauth_storage", mock_storage):
            app = mcp.http_app(transport="streamable-http")
            app.routes.append(Route("/health", health))
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/health")

        assert resp.status_code == 503
        assert resp.json() == {"status": "degraded", "db": "unreachable"}
