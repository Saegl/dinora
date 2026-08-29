"""
The `uci` handshake has to advertise well formed options, `spin` without
`min`/`max` is accepted by python-chess but rejected by fastchess-cli.

`--help` documents those same options, it replaces the hand written
`uci_options_description.txt` that used to be attached to releases.
"""

import subprocess
import sys

import pytest

from dinora.options import uci_options
from dinora.search.registry import DEFAULT_SEARCHER, SEARCHERS, build_searcher


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


def run_cli(*args: str) -> str:
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "dinora", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


def test_help_documents_the_default_searcher() -> None:
    stdout = run_cli("--help")

    assert f"UCI options of searcher `{DEFAULT_SEARCHER}`" in stdout
    assert "move_overhead_ms" in stdout
    assert "batch_size" in stdout
    assert "Options differ per searcher" in stdout


def test_help_documents_the_selected_searcher() -> None:
    stdout = run_cli("--searcher", "ext_mcts", "--help")

    assert "UCI options of searcher `ext_mcts`" in stdout
    assert "fpu_at_root" in stdout
    assert "batch_size" not in stdout  # a mcts_batch only option


def test_searcher_can_follow_help() -> None:
    assert run_cli("--help", "--searcher", "mcts") == run_cli(
        "--searcher", "mcts", "--help"
    )
