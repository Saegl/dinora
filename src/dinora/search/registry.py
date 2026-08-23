"""
Searcher registry.

Values are `"module:attribute"` import specs, not classes, so that importing
this module stays free of numpy & co. See `dinora.threads` for why that matters.
"""

import importlib
from typing import Any

from dinora.search.base import BaseSearcher

DEFAULT_SEARCHER = "mcts_batch"

SEARCHERS: dict[str, str] = {
    "ext_mcts": "dinora.search.ext_mcts.searcher:ExtMcts",
    "mcts": "dinora.search.mcts.mcts:MCTS",
    "mcts_batch": "dinora.search.mcts_batch.mcts_batch:MctsBatch",
    "onemove": "dinora.search.onemove.onemove:OneMove",
}


class UnknownSearcher(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(
            f"Unknown searcher '{name}', available: {', '.join(sorted(SEARCHERS))}"
        )


def build_searcher(name: str | None = None) -> BaseSearcher[Any]:
    name = name or DEFAULT_SEARCHER

    if name not in SEARCHERS:
        raise UnknownSearcher(name)

    module_name, _, attribute = SEARCHERS[name].partition(":")
    cls: type[BaseSearcher[Any]] = getattr(
        importlib.import_module(module_name), attribute
    )
    return cls()
