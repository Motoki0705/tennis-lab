from src.automation.chatgpt_mcp.tunnel import QuickTunnel


def test_extract_public_url() -> None:
    line = "INF +-------------------------------- https://quiet-sun.trycloudflare.com ready"
    assert QuickTunnel.extract_public_url(line) == "https://quiet-sun.trycloudflare.com"


def test_extract_public_url_rejects_other_https_url() -> None:
    assert QuickTunnel.extract_public_url("https://example.com") is None
