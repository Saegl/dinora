"""
The `uci` handshake has to advertise well formed options, `spin` without
`min`/`max` is accepted by python-chess but rejected by fastchess-cli.
"""

import pytest

from dinora.options import uci_options
from dinora.search.registry import SEARCHERS, build_searcher


@pytest.mark.parametrize("searcher_name", sorted(SEARCHERS))
def test_advertised_options_are_well_formed(searcher_name: str) -> None:
    searcher = build_searcher(searcher_name)

    for option in uci_options(searcher.params):
        tokens = option.line().split()
        assert tokens[:3] == ["option", "name", option.name]

        if option.uci_type != "spin":
            continue

        assert "min" in tokens and "max" in tokens, option.line()
        low = int(tokens[tokens.index("min") + 1])
        high = int(tokens[tokens.index("max") + 1])
        assert low <= int(option.default) <= high, option.line()


@pytest.mark.parametrize("searcher_name", sorted(SEARCHERS))
def test_int_params_are_spin_and_float_params_are_string(searcher_name: str) -> None:
    searcher = build_searcher(searcher_name)

    for option in uci_options(searcher.params):
        expected = "spin" if option.value_type is int else "string"
        assert option.uci_type == expected, option.name
