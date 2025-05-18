from typing import Any

from dinora.search.base import BaseSearcher
from dinora.search.ext_mcts.searcher import ExtMcts
from dinora.search.mcts.mcts import MCTS
from dinora.search.mcts_batch.mcts_batch import MctsBatch
from dinora.search.onemove.onemove import OneMove

DEFAULT_SEARCHER: type[BaseSearcher[Any]] = MctsBatch

registered_searchers: dict[str, type[BaseSearcher[Any]]] = {
    "ext_mcts": ExtMcts,
    "mcts": MCTS,
    "mcts_batch": MctsBatch,
    "onemove": OneMove,
}


def get_searcher(searcher_name: str | None) -> BaseSearcher[Any]:
    if searcher_name is None:
        cls = DEFAULT_SEARCHER
    else:
        cls = registered_searchers[searcher_name]
    return cls()
