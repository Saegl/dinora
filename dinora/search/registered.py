from dinora.search.base import BaseSearcher
from dinora.search.ext_mcts.searcher import ExtMcts

registered_searchers: dict[str, type[BaseSearcher]] = {
    "auto": ExtMcts,
    "ext_mcts": ExtMcts,
}


def get_searcher(searcher_name: str) -> BaseSearcher:
    cls = registered_searchers[searcher_name]
    return cls()
