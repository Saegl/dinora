"""
UCI option metadata for searcher params.

Params are plain dataclasses; each field declares its range with `param()` so that
the `uci` handshake, `--help` and value parsing all read the same source.

NOTE: never add `from __future__ import annotations` to a params module. Field
types are read as real objects here, the string form emits zero options.
"""

from dataclasses import dataclass, field, fields
from typing import Any

SUPPORTED_TYPES = (int, float, str)


@dataclass(frozen=True)
class OptionSpec:
    minimum: float | None = None
    maximum: float | None = None
    doc: str = ""


def param(
    *,
    default: Any,
    minimum: float | None = None,
    maximum: float | None = None,
    doc: str = "",
) -> Any:
    """Dataclass field carrying the UCI metadata of a searcher param."""
    return field(default=default, metadata={"uci": OptionSpec(minimum, maximum, doc)})


@dataclass(frozen=True)
class UciOption:
    name: str
    value_type: type
    default: Any
    spec: OptionSpec

    @property
    def uci_type(self) -> str:
        # UCI has no float type, `string` is what lc0 sends for those too
        return "spin" if self.value_type is int else "string"

    def line(self) -> str:
        line = f"option name {self.name} type {self.uci_type} default {self.default}"
        if self.uci_type == "spin":
            assert self.spec.minimum is not None and self.spec.maximum is not None, (
                f"spin option '{self.name}' needs `minimum` and `maximum`"
            )
            line += f" min {int(self.spec.minimum)} max {int(self.spec.maximum)}"
        return line


def uci_options(params: Any) -> list[UciOption]:
    options = []
    for f in fields(params):
        if f.type not in SUPPORTED_TYPES:
            continue

        options.append(
            UciOption(
                name=f.name,
                value_type=f.type,
                default=getattr(params, f.name),
                spec=f.metadata.get("uci", OptionSpec()),
            )
        )
    return options
