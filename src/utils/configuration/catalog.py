"""Source-complete inspectable configuration-contract catalog."""

from __future__ import annotations

from dataclasses import dataclass

import src.utils.configuration.discovery as configuration_discovery
from src.utils.configuration.contracts import (
    RuntimeBoundaryReference,
    discover_boundary_authorities,
    discover_configuration_contracts,
    discover_contract_declarations,
)
from src.utils.paths import PROJECT_ROOT

_SOURCE_ROOT = (PROJECT_ROOT / "src").resolve()
_CANONICAL_SYNTHETIC_MODULES = frozenset(
    {
        "src.synthetic_data_generation.scripts.run_scene_pipeline",
        "src.synthetic_data_generation.scripts.visualize_dataset",
    }
)
SOURCE_CONTRACT_DECLARATIONS = discover_contract_declarations(_SOURCE_ROOT)
ADAPTER_CONTRACTS = discover_configuration_contracts(_SOURCE_ROOT)


@dataclass(frozen=True, slots=True)
class RuntimeBoundaryContract:
    """Inspectable field and semantic authorities for one execution boundary."""

    boundary_id: str
    validator_callable: str
    authority_symbols: tuple[str, ...]
    field_paths: tuple[str, ...]
    semantic_constraint_authorities: tuple[str, ...]
    path_role_authorities: tuple[str, ...]


def _boundary_contracts() -> tuple[RuntimeBoundaryContract, ...]:
    discovered = tuple(
        boundary
        for boundary in configuration_discovery.discover_runtime_boundaries(_SOURCE_ROOT)
        if not boundary.module.startswith("src.synthetic_data_generation.")
        or boundary.module in _CANONICAL_SYNTHETIC_MODULES
    )
    references = tuple(
        RuntimeBoundaryReference(
            boundary_id=f"{boundary.module}:{boundary.callable_name}",
            module=boundary.module,
            callable_name=boundary.callable_name,
            validator_key=boundary.validator_key,
            validator_callable=boundary.validator_callable,
        )
        for boundary in discovered
    )
    bindings = discover_boundary_authorities(
        _SOURCE_ROOT,
        references,
        ADAPTER_CONTRACTS,
    )
    contracts_by_symbol = {
        contract.adapter_symbol: contract for contract in ADAPTER_CONTRACTS
    }
    contracts: list[RuntimeBoundaryContract] = []
    for boundary in discovered:
        boundary_id = f"{boundary.module}:{boundary.callable_name}"
        binding = bindings[boundary_id]
        authorities = tuple(
            contracts_by_symbol[symbol] for symbol in binding.authority_symbols
        )
        validator = boundary.validator_callable or (
            f"{boundary.module}.{boundary.callable_name}"
        )
        semantic_authorities = tuple(
            dict.fromkeys(
                (
                    validator,
                    *binding.semantic_authorities,
                    *(
                        constraint
                        for authority in authorities
                        for constraint in authority.semantic_constraints
                    ),
                )
            )
        )
        contracts.append(
            RuntimeBoundaryContract(
                boundary_id=boundary_id,
                validator_callable=validator,
                authority_symbols=binding.authority_symbols or (validator,),
                field_paths=tuple(
                    f"{authority.adapter_symbol}:{field.path}"
                    for authority in authorities
                    for field in authority.fields
                ),
                semantic_constraint_authorities=semantic_authorities,
                path_role_authorities=tuple(
                    dict.fromkeys(
                        (
                            *binding.path_role_authorities,
                            *(
                                f"{authority.adapter_symbol}:{field.path}:{constraint}"
                                for authority in authorities
                                for field in authority.fields
                                for constraint in field.value_constraints
                                if constraint.startswith("path-role:")
                            ),
                        )
                    )
                ),
            )
        )
    return tuple(contracts)


BOUNDARY_CONTRACTS = _boundary_contracts()

__all__ = [
    "ADAPTER_CONTRACTS",
    "BOUNDARY_CONTRACTS",
    "RuntimeBoundaryContract",
    "SOURCE_CONTRACT_DECLARATIONS",
]
