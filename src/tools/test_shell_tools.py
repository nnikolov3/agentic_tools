import logging
from typing import Any

import httpx
import pytest
from pytest_httpx import HTTPXMock

from src.tools.shell_tools import ShellTools

# Mock configuration for testing
MOCK_CONFIG: dict[str, Any] = {
    "test_agent": {},
    "common_project_files": [],
    "project_directories": [],
    "include_extensions": [],
    "exclude_directories": [],
    "exclude_files": [],
}


@pytest.fixture
def shell_tools():
    """Fixture to provide an instance of ShellTools."""
    return ShellTools("test_agent", MOCK_CONFIG)


@pytest.mark.asyncio
async def test_fetch_urls_content_success(
    shell_tools: ShellTools, httpx_mock: HTTPXMock
):
    """Tests successful fetching of content from multiple URLs."""
    urls = ["http://test.com/page1", "http://test.com/page2"]

    httpx_mock.add_response(url=urls[0], text="Content of page 1.")
    httpx_mock.add_response(url=urls[1], text="Content of page 2.")

    result = await shell_tools.fetch_urls_content(urls)

    expected_output = (
        "\n\n--- Content from: http://test.com/page1 ---\n\nContent of page 1.\n\n"
        "--- Content from: http://test.com/page2 ---\n\nContent of page 2."
    )
    assert result.strip() == expected_output.strip()


@pytest.mark.asyncio
async def test_fetch_urls_content_mixed_results(
    shell_tools: ShellTools, httpx_mock: HTTPXMock, caplog
):
    """Tests fetching with a mix of success, 404, and connection error."""
    urls = [
        "http://test.com/success",
        "http://test.com/not-found",
        "http://test.com/error",
    ]

    httpx_mock.add_response(url=urls[0], text="Success content.")
    httpx_mock.add_response(url=urls[1], status_code=404)

    # Simulate a connection error for the third URL
    async def error_response(request):
        raise httpx.ConnectError("Connection failed")

    httpx_mock.add_callback(error_response, url=urls[2])

    with caplog.at_level(logging.WARNING):
        result = await shell_tools.fetch_urls_content(urls)

    expected_output = (
        "\n\n--- Content from: http://test.com/success ---\n\nSuccess content."
    )
    assert result.strip() == expected_output.strip()

    # Check logs for warnings about failed fetches
    assert (
        "Failed to fetch content from http://test.com/not-found: Status code: 404"
        in caplog.text
    )
    assert (
        "Failed to fetch content from http://test.com/error: Connection failed"
        in caplog.text
    )


@pytest.mark.asyncio
async def test_fetch_urls_content_empty_list(shell_tools: ShellTools):
    """Tests fetching with an empty list of URLs."""
    result = await shell_tools.fetch_urls_content([])
    assert result == ""


@pytest.mark.asyncio
async def test_fetch_urls_content_all_failures(
    shell_tools: ShellTools, httpx_mock: HTTPXMock, caplog
):
    """Tests fetching where all URLs fail."""
    urls = ["http://fail1.com", "http://fail2.com"]

    httpx_mock.add_response(url=urls[0], status_code=500)
    httpx_mock.add_response(url=urls[1], status_code=403)

    with caplog.at_level(logging.WARNING):
        result = await shell_tools.fetch_urls_content(urls)

    assert result == ""
    assert (
        "Failed to fetch content from http://fail1.com: Status code: 500" in caplog.text
    )
    assert (
        "Failed to fetch content from http://fail2.com: Status code: 403" in caplog.text
    )
