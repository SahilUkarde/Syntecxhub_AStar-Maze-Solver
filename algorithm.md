# A* Algorithm — Deep Dive

## Overview

A* (pronounced "A-star") is an informed search algorithm that finds the shortest
path between a start node and a goal node in a weighted graph. It extends Dijkstra's
algorithm by using a heuristic to guide the search toward the goal, making it
significantly faster in practice.

---

## Core Formula

```
f(n) = g(n) + h(n)
```

- **g(n)** — exact cost to reach node `n` from the start
- **h(n)** — heuristic estimate of cost from `n` to the goal
- **f(n)** — total estimated cost; A* always expands the node with lowest f(n)

---

## Algorithm Steps

```
1. Add START to the open list with f = h(START, GOAL)
2. While open list is not empty:
   a. Pop node with lowest f(n) → call it CURRENT
   b. If CURRENT == GOAL → reconstruct and return path
   c. Move CURRENT to closed list
   d. For each NEIGHBOR of CURRENT (not a wall, not in closed list):
      - tentative_g = g(CURRENT) + edge_cost
      - If tentative_g < g(NEIGHBOR):
          * Set came_from[NEIGHBOR] = CURRENT
          * Update g(NEIGHBOR) = tentative_g
          * Update f(NEIGHBOR) = tentative_g + h(NEIGHBOR, GOAL)
          * Add NEIGHBOR to open list
3. If open list is empty → no path exists (UNREACHABLE)
```

---

## Why A* Is Optimal

A* guarantees the shortest path when the heuristic is **admissible**:

> A heuristic h(n) is admissible if it **never overestimates** the true cost.

For a grid with unit step costs:
- **Manhattan** distance is admissible for 4-directional movement
  (it equals the real distance when there are no walls)
- **Euclidean** distance is always admissible (straight line ≤ any actual path)
- **Chebyshev** is admissible when diagonal moves cost 1

---

## Data Structures Used

| Structure | Role |
|-----------|------|
| `heapq` (min-heap) | Open list — always yields lowest-f node in O(log n) |
| `dict` (g_score) | Tracks best known cost to each node |
| `dict` (came_from) | Backtracking map for path reconstruction |
| `set` (closed) | Tracks fully-explored nodes |

---

## Complexity

| Metric | Value |
|--------|-------|
| Time   | O(b^d) worst case; much better with good heuristic |
| Space  | O(b^d) — all nodes stored |
| Optimality | ✅ with admissible heuristic |
| Completeness | ✅ always finds path if exists |

---

## Comparison with Other Algorithms

| Algorithm | Uses heuristic? | Optimal? | Complete? | Notes |
|-----------|----------------|----------|-----------|-------|
| BFS | No | Yes (unweighted) | Yes | Explores in rings |
| Dijkstra | No | Yes | Yes | Like A* with h=0 |
| **A\*** | **Yes** | **Yes** | **Yes** | Best of both worlds |
| Greedy Best-First | Yes | No | No | Fast but not optimal |
| DFS | No | No | No (infinite graphs) | Memory efficient |

---

## Heuristic Trade-offs

### Manhattan Distance
```python
h = abs(r1 - r2) + abs(c1 - c2)
```
- Perfect for 4-directional grids
- Ignores diagonal shortcuts
- Most common choice for maze problems

### Euclidean Distance
```python
h = sqrt((r1-r2)**2 + (c1-c2)**2)
```
- True straight-line distance
- Slightly underestimates on grids (admissible)
- Better when any-angle paths are possible

### Chebyshev Distance
```python
h = max(abs(r1-r2), abs(c1-c2))
```
- Assumes diagonal moves cost 1 (same as cardinal)
- Optimal for 8-directional movement
- Overestimates on 4-directional grids (not admissible there)

---

## Path Reconstruction

Once the goal is reached, the path is built by following `came_from` backwards:

```python
path = [goal]
current = goal
while current in came_from:
    current = came_from[current]
    path.append(current)
path.reverse()  # start → goal
```

---

## Real-World Applications

- **GPS navigation** (Google Maps, Waze)
- **Game AI** (NPC pathfinding in video games)
- **Robotics** (motion planning)
- **Network routing** (packet delivery)
- **Puzzle solving** (15-puzzle, Rubik's cube)
