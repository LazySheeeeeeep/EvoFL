from __future__ import annotations

from typing import Any


FAULT_FAMILIES: tuple[str, ...] = (
    "missing_functionality",
    "state_assignment",
    "validation_control",
    "interface_contract",
    "transformation_representation",
    "algorithm_computation",
    "dispatch_resolution",
    "lifecycle_timing_resource",
    "environment_integration",
)

FAULT_FAMILY_DEFINITIONS: dict[str, str] = {
    "missing_functionality": "Expected capability, handling path, or behavior is absent or incomplete.",
    "state_assignment": "State or values are assigned, synchronized, propagated, or updated incorrectly.",
    "validation_control": "Validation, guards, branching, exception handling, or control flow is incorrect.",
    "interface_contract": "A caller, callee, API, or boundary violates an input, output, or compatibility contract.",
    "transformation_representation": "Parsing, conversion, serialization, normalization, shape, or representation is incorrect.",
    "algorithm_computation": "The core algorithm, calculation, ordering, or logical computation is incorrect.",
    "dispatch_resolution": "The wrong handler, plugin, overload, implementation, or registered target is selected.",
    "lifecycle_timing_resource": "Initialization, cleanup, caching, resource ownership, ordering, or timing is incorrect.",
    "environment_integration": "Dependencies, platforms, external services, configuration environments, or integration boundaries are incompatible.",
}


def validate_fault_family(value: Any) -> str:
    family = str(value or "").strip()
    if family not in FAULT_FAMILIES:
        raise ValueError(
            f"Invalid fault_family {family!r}; expected one of {', '.join(FAULT_FAMILIES)}."
        )
    return family


def compact_fault_taxonomy() -> list[dict[str, str]]:
    return [
        {"fault_family": family, "definition": FAULT_FAMILY_DEFINITIONS[family]}
        for family in FAULT_FAMILIES
    ]


def normalize_retrieval_families(value: Any, primary: str) -> list[str]:
    """A Skill always has its primary entry; aliases share the same identity."""
    primary = validate_fault_family(primary)
    if value is None:
        return [primary]
    if not isinstance(value, list) or not value:
        raise ValueError("retrieval_families must be a non-empty list of fault family IDs.")
    return list(dict.fromkeys([primary, *[validate_fault_family(item) for item in value]]))


def render_fault_taxonomy() -> str:
    return "\n".join(
        f"- {family}: {FAULT_FAMILY_DEFINITIONS[family]}"
        for family in FAULT_FAMILIES
    )
