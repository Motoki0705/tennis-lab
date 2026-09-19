"""Train-only domain exposure balancing, independent of label confidence."""

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import Sampler


@dataclass(frozen=True, slots=True)
class DomainSamplingConfig:
    enabled: bool
    video_domains: dict[str, str]

    def __post_init__(self) -> None:
        for video, domain in self.video_domains.items():
            if not isinstance(video, str) or not video or video != video.strip():
                raise ValueError(
                    "Sampling video IDs must be non-empty trimmed strings."
                )
            if not isinstance(domain, str) or not domain or domain != domain.strip():
                raise ValueError("Sampling domains must be non-empty trimmed strings.")
        if self.enabled and not self.video_domains:
            raise ValueError("Enabled domain sampling requires video_domains.")


class DomainBalancedSampler(Sampler[int]):
    """Inverse domain-count replacement draws, reproducible by seed and epoch.

    Extra mapped videos are allowed, but every configured domain must have a
    retained train window. Public weights/domains and set_epoch support audits.
    """

    def __init__(
        self, video_ids: Sequence[str], video_domains: Mapping[str, str], *, seed: int
    ) -> None:
        DomainSamplingConfig(enabled=True, video_domains=dict(video_domains))
        if not video_ids:
            raise ValueError("Domain sampling requires non-empty train windows.")
        missing = set(video_ids) - video_domains.keys()
        if missing:
            raise ValueError(f"Missing train video domain mappings: {sorted(missing)}")
        self.domains = tuple(video_domains[video] for video in video_ids)
        counts = Counter(self.domains)
        empty = set(video_domains.values()) - counts.keys()
        if empty:
            raise ValueError(
                f"Configured domains have no retained train windows: {sorted(empty)}"
            )
        self.weights = torch.tensor(
            [1.0 / counts[d] for d in self.domains], dtype=torch.double
        )
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("Sampling epoch must be non-negative.")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.domains)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, len(self), replacement=True, generator=generator
        )
        return iter(indices.tolist())
