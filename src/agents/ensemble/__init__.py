"""Ensemble functionality for consulting multiple LLM providers."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from src.agents.providers.base import Provider, ProviderRequest, ProviderResult
from src.agents.providers.registry import get_provider


@dataclass
class ConsultationResult:
    """Result of consulting multiple providers."""

    results: list[ProviderResult] = field(default_factory=list)

    @property
    def successful_results(self) -> list[ProviderResult]:
        """Get only successful results."""
        return [r for r in self.results if r.success]

    @property
    def failed_results(self) -> list[ProviderResult]:
        """Get only failed results."""
        return [r for r in self.results if not r.success]

    def format_for_context(self) -> str:
        """Format results for insertion into context."""
        lines = ["=" * 60, "Sub-agent Consultation Results", "=" * 60, ""]

        for result in self.results:
            status = "✓" if result.success else "✗"
            lines.append(f"[{result.provider.value.upper()}] {status}")
            lines.append("-" * 40)

            if result.success:
                lines.append(result.output.strip())
            else:
                lines.append(f"Error: {result.error}")

            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


class Ensemble:
    """Ensemble of LLM providers for parallel consultation."""

    def __init__(self, providers: list[Provider]):
        """Initialize with list of providers to use."""
        self.providers = providers

    def consult_parallel(
        self,
        request: ProviderRequest,
        max_workers: int = 4,
    ) -> ConsultationResult:
        """Consult all providers in parallel."""
        results: list[ProviderResult] = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for provider in self.providers:
                runner = get_provider(provider)
                future = executor.submit(runner.run, request)
                futures[future] = provider

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    provider = futures[future]
                    results.append(
                        ProviderResult(
                            provider=provider,
                            success=False,
                            error=str(e),
                        )
                    )

        return ConsultationResult(results=results)

    def consult_sequential(self, request: ProviderRequest) -> ConsultationResult:
        """Consult all providers sequentially."""
        results: list[ProviderResult] = []

        for provider in self.providers:
            runner = get_provider(provider)
            result = runner.run(request)
            results.append(result)

        return ConsultationResult(results=results)
