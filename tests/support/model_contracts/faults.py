"""Reversible named-head faults shared by task registrations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tests.support.model_contracts.parameters import names_under


def head_fault(path: str, kind: str) -> Callable[[Any], None]:
    def inject(module: Any) -> None:
        head = module.get_submodule(path)
        if kind == "detach":
            head.register_forward_hook(lambda _module, _args, output: output.detach())
        elif kind == "optimizer":
            excluded = names_under(module, path)
            original = module.optimizer_param_groups

            def groups() -> list[dict[str, Any]]:
                declared = original()
                if declared is None:
                    declared = [{"params": module.parameters()}]
                by_id = {id(p): name for name, p in module.named_parameters()}
                return [
                    {
                        **group,
                        "params": [
                            p for p in group["params"] if by_id[id(p)] not in excluded
                        ],
                    }
                    for group in declared
                ]

            module.optimizer_param_groups = groups
        else:
            raise ValueError(f"Unknown head fault: {kind}")

    return inject
