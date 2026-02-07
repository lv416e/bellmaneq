"""Game board visualization."""

import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle


def plot_tictactoe(
    board: list[int],
    *,
    title: str = "Tic-Tac-Toe",
    values: list[float] | None = None,
    ax: plt.Axes | None = None,
) -> Figure | None:
    """Renders a Tic-Tac-Toe board.

    Args:
        board: Length-9 list (0 = empty, 1 = X, 2 = O).
        title: Plot title.
        values: Per-cell minimax values (displayed as a heatmap).
        ax: Existing Axes to draw on.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14)

    # Grid lines
    for i in range(4):
        ax.axhline(i - 0.5, color="black", linewidth=1)
        ax.axvline(i - 0.5, color="black", linewidth=1)

    # Value heatmap
    if values is not None:
        vals = np.array(values).reshape(3, 3)
        ax.imshow(
            vals[::-1],
            extent=(-0.5, 2.5, -0.5, 2.5),
            cmap="RdYlGn",
            alpha=0.3,
            vmin=-1,
            vmax=1,
        )

    # Draw pieces
    for i, cell in enumerate(board):
        row, col = divmod(i, 3)
        y = 2 - row  # Convert top-down row index to bottom-up coordinate
        if cell == 1:  # X
            ax.plot(
                [col - 0.3, col + 0.3],
                [y - 0.3, y + 0.3],
                "b-",
                linewidth=3,
            )
            ax.plot(
                [col - 0.3, col + 0.3],
                [y + 0.3, y - 0.3],
                "b-",
                linewidth=3,
            )
        elif cell == 2:  # O
            circle = Circle((col, y), 0.3, fill=False, color="red", linewidth=3)
            ax.add_patch(circle)

    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def plot_connect_four(
    board: list[list[int]],
    *,
    title: str = "Connect Four",
    last_move: tuple[int, int] | None = None,
    ax: plt.Axes | None = None,
) -> Figure | None:
    """Renders a Connect Four board.

    Args:
        board: 6x7 2D list (0 = empty, 1 = red, 2 = yellow).
        title: Plot title.
        last_move: (row, col) of the last move to highlight.
        ax: Existing Axes to draw on.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 7))

    rows = len(board)
    cols = len(board[0]) if board else 7

    # Blue board background
    bg = Rectangle((-0.5, -0.5), cols, rows, facecolor="#1565C0", zorder=0)
    ax.add_patch(bg)

    for r in range(rows):
        for c in range(cols):
            cell = board[r][c]
            y = rows - 1 - r  # Screen position

            if cell == 0:
                color = "white"
            elif cell == 1:
                color = "#F44336"  # Red
            else:
                color = "#FFEB3B"  # Yellow

            circle = Circle(
                (c, y),
                0.4,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
                zorder=1,
            )
            ax.add_patch(circle)

            # Highlight the last move
            if last_move and last_move == (r, c):
                highlight = Circle(
                    (c, y),
                    0.42,
                    fill=False,
                    edgecolor="white",
                    linewidth=3,
                    zorder=2,
                )
                ax.add_patch(highlight)

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14)
    ax.set_xticks(range(cols))
    ax.set_xticklabels([str(i) for i in range(cols)])
    ax.set_yticks([])

    return fig


def plot_minimax_tree(
    game,
    board: list,
    next_player: int,
    depth: int = 3,
    *,
    title: str = "Minimax Search Tree",
) -> Figure:
    """Renders a minimax search tree using matplotlib.

    Builds the game tree to the specified depth, annotates each node
    with its minimax value, and highlights the principal variation (PV).
    Max-player nodes are drawn as blue upward triangles; min-player nodes
    as red downward triangles.

    Args:
        game: Game engine with ``minimax(board, player, depth)`` and
              ``legal_actions(board, player)`` methods.
        board: Current board state.
        next_player: Player to move (1 or 2).
        depth: Maximum tree depth to explore.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    # -- Build the tree --------------------------------------------------
    nodes: dict[int, dict] = {}  # id -> {value, children, is_max, action}
    counter = [0]

    def _build(b, player, d):
        nid = counter[0]
        counter[0] += 1
        is_max = player == 1
        value = game.minimax(b, player, d)
        children: list[tuple[int, int]] = []  # (action, child_id)

        if d > 0:
            actions = game.legal_actions(b, player)
            opp = 2 if player == 1 else 1
            for a in actions:
                if hasattr(game, "apply_move"):
                    child_board = game.apply_move(b, player, a)
                else:
                    child_board = list(b)
                    child_board[a] = player
                child_id = _build(child_board, opp, d - 1)
                children.append((a, child_id))

        nodes[nid] = {
            "value": value,
            "children": children,
            "is_max": is_max,
            "action": None,
        }
        return nid

    if len(nodes) == 0 and counter[0] < 1500:
        _build(board, next_player, depth)
    else:
        _build(board, next_player, depth)

    if counter[0] > 1500:
        warnings.warn(
            f"Tree has {counter[0]} nodes; rendering may be slow. "
            "Consider reducing depth.",
            stacklevel=2,
        )

    # -- Assign positions ------------------------------------------------
    pos: dict[int, tuple[float, float]] = {}
    leaf_counter = [0]

    def _layout(nid, level):
        children = nodes[nid]["children"]
        if not children:
            x = leaf_counter[0]
            leaf_counter[0] += 1
            pos[nid] = (x, -level)
            return
        for _, cid in children:
            _layout(cid, level + 1)
        xs = [pos[cid][0] for _, cid in children]
        pos[nid] = ((min(xs) + max(xs)) / 2.0, -level)

    _layout(0, 0)

    # -- Find principal variation (PV) ------------------------------------
    pv_edges: set[tuple[int, int]] = set()

    def _find_pv(nid):
        children = nodes[nid]["children"]
        if not children:
            return
        is_max = nodes[nid]["is_max"]
        best_child = None
        best_val = float("-inf") if is_max else float("inf")
        for _, cid in children:
            cv = nodes[cid]["value"]
            if (is_max and cv > best_val) or (not is_max and cv < best_val):
                best_val = cv
                best_child = cid
        if best_child is not None:
            pv_edges.add((nid, best_child))
            _find_pv(best_child)

    _find_pv(0)

    # -- Render ----------------------------------------------------------
    n_leaves = leaf_counter[0]
    fig_w = max(8, n_leaves * 0.6)
    fig_h = max(4, (depth + 1) * 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Edges
    for nid, data in nodes.items():
        for _, cid in data["children"]:
            x0, y0 = pos[nid]
            x1, y1 = pos[cid]
            is_pv = (nid, cid) in pv_edges
            ax.plot(
                [x0, x1],
                [y0, y1],
                color="#FFB300" if is_pv else "#BDBDBD",
                linewidth=2.5 if is_pv else 0.8,
                zorder=0,
            )

    # Nodes
    for nid, data in nodes.items():
        x, y = pos[nid]
        marker = "^" if data["is_max"] else "v"
        color = "#1976D2" if data["is_max"] else "#D32F2F"
        ax.scatter(x, y, marker=marker, s=120, c=color, zorder=2, edgecolors="white", linewidths=0.5)
        ax.annotate(
            f"{data['value']:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10 if data["is_max"] else -14),
            ha="center",
            fontsize=6,
            color=color,
        )

    ax.set_title(title, fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    return fig
