"""
Visualization of MCTS
Look at jupyter/treeviz.ipynb for an example of using this module
"""
from dataclasses import dataclass
from time import sleep

import graphviz
import cairosvg

import chess.svg
from dinora.search import *


@dataclass
class RenderParams:
    max_number_of_nodes: int = 150
    show_other_node: bool = True
    show_prior = True


def get_all_nodes(node: UCTNode):
    """Get all reachable nodes from a given node"""
    for _, child in node.children.items():
        yield child
        yield from get_all_nodes(child)


def select_best_nodes(node: UCTNode, n: int) -> set[UCTNode]:
    """Get n best nodes from a given node, sorted by number of visits"""
    all_nodes = list(get_all_nodes(node))
    all_nodes.sort(reverse=True, key=lambda e: e.number_visits)
    return set(all_nodes[:n])


def build_root_node(dot: graphviz.Digraph, fen: str):
    """Save root node in digraph and attach board preview"""
    # FIXME: cairo is big dependency, is there better way to convert svg to png?
    cairosvg.svg2png(chess.svg.board(chess.Board(fen)), write_to="generated/cboard.png")
    dot.node("root", label="", image="cboard.png", imagescale="false", shape="box")


def build_children_nodes(
    dot: graphviz.Digraph,
    node: UCTNode,
    parent_id: str,
    selected_nodes: set[UCTNode],
    params: RenderParams,
):
    others_visits = 0
    others_id = str(id(node.children))
    others_prior = 0.0
    for move, child in node.children.items():
        if child not in selected_nodes or child.number_visits == 0:
            others_visits += child.number_visits
            others_prior += child.prior
            continue

        child_id = str(id(child))
        label = f"{{ {str(move)} | Q:{child.Q():.2f} N:{child.number_visits} }}"

        dot.node(child_id, label, shape="record")
        prior_props = {"label": f"{child.prior:.2f}"} if params.show_prior else {}
        dot.edge(parent_id, child_id, **prior_props)
        build_children_nodes(dot, child, child_id, selected_nodes, params)

    others_label = f"{{ othr | Q:?.?? N:{others_visits} }}"

    if params.show_other_node:
        dot.node(others_id, others_label, shape="record", color="orange")
        prior_props = {"label": f"{others_prior:.2f}"} if params.show_prior else {}
        dot.edge(parent_id, others_id, **prior_props)


def build_graph(
    root: UCTNode,
    fen=chess.STARTING_BOARD_FEN,
    format="png",
    params: RenderParams = RenderParams(),
) -> graphviz.Digraph:

    dot = graphviz.Digraph(
        "search-tree",
        format=format,
        comment="Monte Carlo Search Tree",
        graph_attr={
            "rankdir": "LR",
            # "bgcolor": "#00000000",
            # "splines": "line",  # FIXME: can you make beautiful straight lines?
        },
    )

    selected_nodes = select_best_nodes(root, params.max_number_of_nodes)

    build_root_node(dot, fen)
    build_children_nodes(dot, root, "root", selected_nodes, params)

    return dot


def render_search_process(
    model,
    fen: str,
    c: float,
    nodes=10,
    sleep_between_states: float = 0.0,
    params: RenderParams = RenderParams(),
):
    for i in range(1, nodes):
        root: UCTNode = uct_nodes(chess.Board(fen), i, model, c, None, 1.0, 0.0, 1.0)
        graph = build_graph(root, params=params, fen=fen)
        graph.render(directory="generated", filename=str(i), view=False)
        sleep(sleep_between_states)


def render_state(
    model,
    fen: str,
    nodes: int,
    c: float,
    format: str = "svg",
    params: RenderParams = RenderParams(),
) -> graphviz.Digraph:
    root: UCTNode = uct_nodes(chess.Board(fen), nodes, model, c, None, 1.0, 0.0, 1.0)
    graph = build_graph(root, params=params, fen=fen, format=format)
    graph.render(directory="generated", filename="state", view=True)
