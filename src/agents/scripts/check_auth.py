"""Check authentication status for all LLM providers.

Example commands:
    `uv run python -m src.agents.scripts.check_auth`

This script checks the authentication status for all supported LLM providers
(Claude, Gemini, Codex, Copilot) and displays the results.
"""

from src.agents.providers import Provider, check_auth, get_provider


def main() -> None:
    """Check and display authentication status for all providers."""
    print("=" * 60)
    print("LLM Provider Authentication Status")
    print("=" * 60)
    print()

    authenticated_providers: list[str] = []

    for provider in Provider:
        runner = get_provider(provider)
        binary_available = runner.check_binary()
        is_auth, auth_method = check_auth(provider)

        print(provider.value.upper())
        print("-" * 40)
        print(f"  Binary available: {'✓' if binary_available else '✗'}")
        print(f"  Authenticated:    {'✓' if is_auth else '✗'}")

        if is_auth:
            print(f"  Auth method:      {auth_method}")
            authenticated_providers.append(provider.value)
        else:
            print(f"  Notes:            {auth_method}")

        print()

    print("=" * 60)
    if authenticated_providers:
        print(f"Ready to use: {', '.join(authenticated_providers)}")
    else:
        print("No providers authenticated")
    print("=" * 60)


if __name__ == "__main__":
    main()
