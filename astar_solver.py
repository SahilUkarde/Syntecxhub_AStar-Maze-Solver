"""
A* Maze Solver
==============
Implements A* search algorithm on a 2D grid maze.
Supports Manhattan, Euclidean, and Chebyshev heuristics.
Includes optional matplotlib visualization.

Author: [Your Name]
Internship Project - 2024
"""

import heapq
import math
import time
from typing import List, Tuple, Optional, Dict

# Cell type constants
EMPTY = 0
WALL  = 1
START = 2
GOAL  = 3
OPEN_SET  = 4
CLOSED_SET = 5
PATH  = 6


# ── Heuristics ──────────────────────────────────────────────────────────────

def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Manhattan distance — optimal for 4-directional grids."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Euclidean (straight-line) distance."""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def chebyshev(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Chebyshev distance — optimal when diagonal moves are allowed."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

HEURISTICS = {
    "manhattan": manhattan,
    "euclidean": euclidean,
    "chebyshev": chebyshev,
}


# ── A* Core ─────────────────────────────────────────────────────────────────

class AStarSolver:
    """
    A* pathfinding on a 2D grid.

    Parameters
    ----------
    grid       : 2D list of ints (0=empty, 1=wall)
    start      : (row, col) tuple
    goal       : (row, col) tuple
    heuristic  : 'manhattan' | 'euclidean' | 'chebyshev'
    allow_diag : whether diagonal moves are permitted
    """

    def __init__(
        self,
        grid: List[List[int]],
        start: Tuple[int, int],
        goal: Tuple[int, int],
        heuristic: str = "manhattan",
        allow_diag: bool = False,
    ):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows else 0
        self.start = start
        self.goal = goal
        self.h = HEURISTICS.get(heuristic, manhattan)
        self.allow_diag = allow_diag

        # Stats
        self.nodes_explored = 0
        self.path: List[Tuple[int, int]] = []
        self.visited_order: List[Tuple[int, int]] = []  # for visualization

    def _neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        directions = [(-1,0),(1,0),(0,-1),(0,1)]
        if self.allow_diag:
            directions += [(-1,-1),(-1,1),(1,-1),(1,1)]
        result = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.grid[nr][nc] != WALL:
                    result.append((nr, nc))
        return result

    def solve(self) -> Optional[List[Tuple[int, int]]]:
        """
        Run A* search.

        Returns the path as a list of (row, col) tuples from start to goal,
        or None if the goal is unreachable.
        """
        start_time = time.perf_counter()

        open_heap: List[Tuple[float, int, Tuple[int,int]]] = []
        counter = 0  # tie-breaker

        g_score: Dict[Tuple[int,int], float] = {self.start: 0}
        f_score: Dict[Tuple[int,int], float] = {self.start: self.h(self.start, self.goal)}
        came_from: Dict[Tuple[int,int], Tuple[int,int]] = {}
        in_open: set = {self.start}

        heapq.heappush(open_heap, (f_score[self.start], counter, self.start))

        while open_heap:
            _, _, current = heapq.heappop(open_heap)

            if current in in_open:
                in_open.discard(current)

            self.nodes_explored += 1
            self.visited_order.append(current)

            if current == self.goal:
                self.path = self._reconstruct(came_from, current)
                elapsed = time.perf_counter() - start_time
                self._print_summary(elapsed)
                return self.path

            for neighbor in self._neighbors(*current):
                tentative_g = g_score[current] + 1

                if tentative_g < g_score.get(neighbor, math.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.h(neighbor, self.goal)
                    f_score[neighbor] = f
                    counter += 1
                    heapq.heappush(open_heap, (f, counter, neighbor))
                    in_open.add(neighbor)

        # Goal unreachable
        elapsed = time.perf_counter() - start_time
        print(f"\n[A*] Goal is UNREACHABLE. Explored {self.nodes_explored} nodes in {elapsed*1000:.1f}ms.")
        return None

    def _reconstruct(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _print_summary(self, elapsed: float):
        print("\n" + "="*45)
        print("  A* Search — Results")
        print("="*45)
        print(f"  Status        : Path found ✓")
        print(f"  Path length   : {len(self.path) - 1} steps")
        print(f"  Nodes explored: {self.nodes_explored}")
        print(f"  Heuristic     : {self.h.__name__}")
        print(f"  Time elapsed  : {elapsed*1000:.2f} ms")
        print("="*45)


# ── Console Visualizer ───────────────────────────────────────────────────────

CELL_CHARS = {
    EMPTY: "  ",
    WALL:  "██",
    START: " S",
    GOAL:  " G",
    OPEN_SET:   " ·",
    CLOSED_SET: " ×",
    PATH:  " *",
}

def print_grid(grid, path=None, visited=None, start=None, goal=None):
    """Print the maze in the console with ASCII art."""
    rows = len(grid)
    cols = len(grid[0])
    display = [[CELL_CHARS.get(grid[r][c], "  ") for c in range(cols)] for r in range(rows)]

    if visited:
        for (r, c) in visited:
            if grid[r][c] == EMPTY:
                display[r][c] = CELL_CHARS[CLOSED_SET]

    if path:
        for (r, c) in path:
            if grid[r][c] == EMPTY:
                display[r][c] = CELL_CHARS[PATH]

    if start:
        display[start[0]][start[1]] = CELL_CHARS[START]
    if goal:
        display[goal[0]][goal[1]] = CELL_CHARS[GOAL]

    border = "+" + "──" * cols + "+"
    print(border)
    for row in display:
        print("|" + "".join(row) + "|")
    print(border)


# ── Matplotlib Visualizer ────────────────────────────────────────────────────

def visualize(grid, path=None, visited=None, start=None, goal=None, title="A* Maze Solver"):
    """
    Render the maze using matplotlib.
    Requires: pip install matplotlib numpy
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
    except ImportError:
        print("[!] matplotlib not installed. Run: pip install matplotlib numpy")
        return

    rows = len(grid)
    cols = len(grid[0])
    display = np.zeros((rows, cols))

    for r in range(rows):
        for c in range(cols):
            display[r][c] = grid[r][c]

    if visited:
        for (r, c) in visited:
            if grid[r][c] == EMPTY:
                display[r][c] = CLOSED_SET

    if path:
        for (r, c) in path:
            if grid[r][c] == EMPTY:
                display[r][c] = PATH

    if start:
        display[start[0]][start[1]] = START
    if goal:
        display[goal[0]][goal[1]] = GOAL

    # Color map: empty, wall, start, goal, open, closed, path
    cmap = mcolors.ListedColormap([
        "#f8fafc",  # 0 empty
        "#334155",  # 1 wall
        "#22c55e",  # 2 start
        "#ef4444",  # 3 goal
        "#bfdbfe",  # 4 open
        "#c7d2fe",  # 5 closed
        "#fbbf24",  # 6 path
    ])
    bounds = [0, 1, 2, 3, 4, 5, 6, 7]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(max(8, cols//2), max(6, rows//2)))
    ax.imshow(display, cmap=cmap, norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#22c55e", label="Start (S)"),
        Patch(facecolor="#ef4444", label="Goal (G)"),
        Patch(facecolor="#334155", label="Wall"),
        Patch(facecolor="#c7d2fe", label="Explored (closed)"),
        Patch(facecolor="#fbbf24", label="Shortest path"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              bbox_to_anchor=(1.18, 1), framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig("maze_result.png", dpi=150, bbox_inches="tight")
    print("[✓] Saved visualization to maze_result.png")
    plt.show()


# ── Example Mazes ────────────────────────────────────────────────────────────

def make_sample_maze():
    """Returns a hand-crafted 10×15 maze with a clear path."""
    W, E = WALL, EMPTY
    maze = [
        [E,E,E,E,W,E,E,E,E,E,E,E,E,E,E],
        [W,W,E,W,W,E,W,W,W,E,W,W,W,W,E],
        [E,E,E,E,E,E,E,E,W,E,E,E,E,W,E],
        [E,W,W,W,W,W,W,E,W,W,W,E,W,W,E],
        [E,W,E,E,E,E,W,E,E,E,E,E,E,E,E],
        [E,W,E,W,W,E,W,W,W,W,W,W,W,W,E],
        [E,W,E,W,E,E,E,E,E,E,E,E,E,W,E],
        [E,E,E,W,E,W,W,W,W,W,E,W,E,W,E],
        [W,W,E,W,E,E,E,E,E,W,E,W,E,E,E],
        [E,E,E,E,E,W,W,W,E,W,E,E,E,W,E],
    ]
    return maze, (0, 0), (9, 14)


def make_open_maze(rows=15, cols=20):
    """Returns an open grid (no walls) for baseline testing."""
    maze = [[EMPTY] * cols for _ in range(rows)]
    return maze, (0, 0), (rows-1, cols-1)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A* Maze Solver")
    parser.add_argument("--heuristic", choices=["manhattan","euclidean","chebyshev"],
                        default="manhattan", help="Heuristic function")
    parser.add_argument("--maze", choices=["sample","open"], default="sample",
                        help="Which maze to solve")
    parser.add_argument("--visualize", action="store_true",
                        help="Show matplotlib visualization")
    parser.add_argument("--diag", action="store_true",
                        help="Allow diagonal movement")
    args = parser.parse_args()

    if args.maze == "sample":
        grid, start, goal = make_sample_maze()
    else:
        grid, start, goal = make_open_maze()

    print(f"\n[A*] Heuristic: {args.heuristic.capitalize()}")
    print(f"[A*] Grid: {len(grid)}×{len(grid[0])}  Start: {start}  Goal: {goal}\n")

    solver = AStarSolver(grid, start, goal,
                         heuristic=args.heuristic,
                         allow_diag=args.diag)
    path = solver.solve()

    print("\n── Console Map ──")
    print_grid(grid, path=path, visited=solver.visited_order, start=start, goal=goal)
    print("\nLegend:  S=Start  G=Goal  *=Path  ×=Explored  ██=Wall\n")

    if args.visualize:
        visualize(grid, path=path, visited=solver.visited_order,
                  start=start, goal=goal,
                  title=f"A* Maze Solver — {args.heuristic.capitalize()} Heuristic")
