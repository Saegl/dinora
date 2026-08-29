"""
UCI option metadata for searcher params.

Params are plain dataclasses; each field declares its range with `param()` so that
the `uci` handshake, `--help` and value parsing all read the same source.

NOTE: never add `from __future__ import annotations` to a params module. Field
types are read as real objects here, the string form emits zero options.
"""

import textwrap
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


def type_label(option: UciOption) -> str:
    return "string" if option.value_type is str else option.value_type.__name__


def render_options(options: list[UciOption], width: int = 88) -> str:
    """Format options as an indented `--help` block."""
    if not options:
        return "  (this searcher has no options)"

    name_width = max(len(option.name) for option in options)
    type_width = max(len(type_label(option)) for option in options)
    default_width = max(len(str(option.default)) for option in options)

    lines = []
    for option in options:
        head = (
            f"  {option.name:<{name_width}}  {type_label(option):<{type_width}}"
            f"  default {str(option.default):<{default_width}}"
        )
        if option.spec.minimum is not None and option.spec.maximum is not None:
            head += f"  range {option.spec.minimum} .. {option.spec.maximum}"
        lines.append(head.rstrip())

        if option.spec.doc:
            lines.append(
                textwrap.fill(
                    option.spec.doc,
                    width=width,
                    initial_indent="      ",
                    subsequent_indent="      ",
                )
            )

    return "\n".join(lines)
