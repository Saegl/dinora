"""
Visualization of MCTS
Look at jupyter/treeviz.ipynb for an example of using this module
"""
from typing import Iterator, Literal
from dataclasses import dataclass

import graphviz

import chess.svg
from dinora import PROJECT_ROOT
from dinora.mcts import run_mcts, MCTSparams, Node, NodesCountConstraint
from dinora.models.base import BaseModel

NodeID = str
SVG_PREFIX = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>'
OUTPUT_DIR = PROJECT_ROOT / "data/treeviz"


def node_id(node: object) -> NodeID:
    return str(id(node))


@dataclass
class RenderParams:
    max_number_of_nodes: int = 150
    show_other_node: bool = True
    show_prior: bool = True
    open_default_gui: bool = False


def get_all_nodes(node: Node) -> Iterator[Node]:
    """Get all reachable nodes from a given node"""
    for _, child in node.children.items():
        yield child
        yield from get_all_nodes(child)


def select_most_visited_nodes(node: Node, n: int) -> set[NodeID]:
    """Select n most visited nodes"""
    all_nodes = list(get_all_nodes(node))
    all_nodes.sort(reverse=True, key=lambda e: e.number_visits)
    return set(map(node_id, all_nodes[:n]))


def tree_shape(root: Node) -> list[int]:
    """Calculate number of nodes on each depth by using dfs"""

    depthlist: list[int] = []

    def dfs(node: Node, depth: int) -> None:
        for _, child in node.children.items():
            if len(depthlist) == depth:
                depthlist.append(0)
            if not child.children:
                continue
            depthlist[depth] += 1
            dfs(child, depth + 1)

    dfs(root, 0)
    return depthlist


def get_pv_set(root: Node) -> set[NodeID]:
    ans = set()

    curr = root
    while len(curr.children) != 0:
        bestchild = curr.best_child(0.0)
        ans.add(node_id(bestchild))
        curr = bestchild
    return ans


def build_info_node(graph: graphviz.Digraph, root: Node) -> None:
    children = list(root.children.items())
    children.sort(key=lambda t: (t[1].number_visits, t[1].Q()), reverse=True)
    top_visited_labels = [
        f"{t[0]} - N:{t[1].number_visits}, Q:{t[1].Q():.3f}" for t in children[:5]
    ]
    top_visited = "\n".join(top_visited_labels)

    children.sort(key=lambda t: (t[1].Q(), t[1].number_visits), reverse=True)
    top_q_highest_labels = [
        f"{t[0]} - Q:{t[1].Q():.3f}, N{t[1].number_visits}" for t in children[:5]
    ]
    top_q = "\n".join(top_q_highest_labels)

    bestchild = root.best_child(0.0)
    info = (
        f"Side to move: {'white' if root.board.turn else 'black'} \n"
        f"Q at bestchild {bestchild.Q():.3f} \n\n"
        f"TOP 5 most visited nodes at root: \n{top_visited} \n\n"
        f"TOP 5 highest Q nodes at root: \n{top_q} \n\n"
        f"Tree Shape: {tree_shape(root)}"
    )
    graph.node("info", label=info, shape="box")


def build_root_node(graph: graphviz.Digraph, fen: str) -> None:
    """Save root node in digraph and attach board preview"""
    svg = SVG_PREFIX + chess.svg.board(chess.Board(fen))
    svg_filename = "svg_board.svg"
    board_path = OUTPUT_DIR / svg_filename
    with open(board_path, "wt", encoding="utf8") as f:
        f.write(svg)
    graph.node("root", label="", image=svg_filename, imagescale="false", shape="box")


def build_children_nodes(
    graph: graphviz.Digraph,
    node: Node,
    parent_id: str,
    selected_nodes: set[NodeID],
    pv_set: set[NodeID],
    params: RenderParams,
) -> None:
    others_visits = 0
    others_id = node_id(node.children)
    others_prior = 0.0
    for move, child in node.children.items():
        child_id = node_id(child)

        if child_id not in selected_nodes or child.number_visits == 0:
            others_visits += child.number_visits
            others_prior += child.prior
            continue

        label = f"{{ {str(move)} | Q:{child.Q():.3f} N:{child.number_visits} VE: {child.board_value_estimate_info:.3f} }}"

        if child_id in pv_set:
            color = "red"
        else:
            color = None
        graph.node(child_id, label, shape="record", color=color)
        prior_props = {"label": f"{child.prior:.3f}"} if params.show_prior else {}
        graph.edge(parent_id, child_id, **prior_props)
        build_children_nodes(graph, child, child_id, selected_nodes, pv_set, params)

    others_label = f"{{ othr | Q:?.?? N:{others_visits} }}"

    if params.show_other_node:
        graph.node(others_id, others_label, shape="record", color="orange")
        prior_props = {"label": f"{others_prior:.2f}"} if params.show_prior else {}
        graph.edge(parent_id, others_id, **prior_props)


def build_graph(
    root: Node,
    fen: str = chess.STARTING_BOARD_FEN,
    format: Literal["png", "svg"] = "png",
    params: RenderParams = RenderParams(),
) -> graphviz.Digraph:
    graph = graphviz.Digraph(
        "search-tree",
        format=format,
        comment="Monte Carlo Search Tree",
        graph_attr={
            "rankdir": "LR",
            # "bgcolor": "#00000000",
            "splines": "line",  # FIXME: can you make beautiful straight lines?
            "size": "150,150",
        },
    )

    pv_set = get_pv_set(root)
    selected_nodes = select_most_visited_nodes(root, params.max_number_of_nodes).union(
        pv_set
    )

    build_info_node(graph, root)
    build_root_node(graph, fen)
    build_children_nodes(graph, root, "root", selected_nodes, pv_set, params)

    return graph


def render_state(
    model: BaseModel,
    fen: str,
    nodes: int,
    format: Literal["png", "svg"] = "svg",
    mcts_params: MCTSparams = MCTSparams(),
    render_params: RenderParams = RenderParams(),
) -> Node:
    board = chess.Board(fen)
    root = run_mcts(board, NodesCountConstraint(nodes), model, mcts_params)
    graph = build_graph(root, params=render_params, fen=fen, format=format)
    graph.render(
        directory=OUTPUT_DIR, filename="state", view=render_params.open_default_gui
    )
    return root
