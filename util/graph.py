from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable
from functools import lru_cache
from typing import (
    DefaultDict,
    Dict,
    Generic,
    List,
    Set,
    Tuple,
    TypeVar,
)

T = TypeVar("T", bound=Hashable)


class UnionFind(Generic[T]):
    """
    Disjoint Set Union (Union–Find) structure for tracking connected components.

    - Each element belongs to exactly one set (component).
    - `find(x)` returns the representative (root) of x's component.
    - `union(a, b)` merges the components of a and b.

    Uses:
    - Path compression in `find`
    - Union by size in `union`
    """

    def __init__(self) -> None:
        self.parent: Dict[T, T] = {}
        self.size: Dict[T, int] = {}

    def add(self, x: T) -> None:
        """Ensure that element `x` is known to the structure."""
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1

    def find(self, x: T) -> T:
        """Return the representative of the set containing `x`."""
        # Path compression
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: T, b: T) -> bool:
        """
        Merge the sets of `a` and `b`.
        """
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False  # already connected

        # Union by size: attach smaller tree to larger
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True

    def in_same_set(self, a: T, b: T) -> bool:
        """Return True if `a` and `b` are in the same component."""
        return self.find(a) == self.find(b)


class Graph(Generic[T]):
    """
    Simple undirected graph with integrated Union–Find for connectivity.

    Attributes
    ----------
    nodes : set[T]
        All nodes in the graph.
    edges : set[tuple[T, T]]
        Set of undirected edges (a, b) as given when added.
    adjacency : dict[T, set[T]]
        Adjacency list; for each node, its set of neighbors.
    uf : UnionFind[T]
        Union–Find structure tracking which nodes are in the same component.

    Notes
    -----
    - No parallel edges are stored (edges are in a set).
    - The graph is undirected: adjacency is symmetric, even if edge tuple
      is stored as (a, b) in `edges`.
    """

    def __init__(self) -> None:
        self.nodes: Set[T] = set()
        self.edges: Set[Tuple[T, T]] = set()
        self.adjacency: DefaultDict[T, Set[T]] = defaultdict(set)
        self.uf: UnionFind[T] = UnionFind()

    def add_node(self, node: T) -> None:
        """
        Add a single node to the graph.

        Also registers the node in the Union–Find structure.
        """
        if node not in self.nodes:
            self.nodes.add(node)
            self.uf.add(node)

    def add_edge(self, pair: Tuple[T, T]) -> None:
        """
        Add an undirected edge between two nodes.

        Both nodes are added if not already present. Connectivity information
        is maintained using Union–Find.

        """
        a, b = pair
        self.add_node(a)
        self.add_node(b)

        # Store edge and update adjacency
        self.edges.add((a, b))
        self.adjacency[a].add(b)
        self.adjacency[b].add(a)

        # Update connectivity structure
        self.uf.union(a, b)

    def find_connected_components(self) -> List[Set[T]]:
        """
        Find all connected components in the graph.

        Uses the Union–Find structure rather than BFS/DFS, so it stays
        efficient even if you call it frequently.
        """
        components: DefaultDict[T, Set[T]] = defaultdict(set)

        for node in self.nodes:
            root = self.uf.find(node)
            components[root].add(node)

        return list(components.values())

    def are_connected(self, a: T, b: T) -> bool:
        """
        Check whether two nodes are in the same connected component.
        """
        if a not in self.nodes or b not in self.nodes:
            return False
        return self.uf.in_same_set(a, b)

    def count_cycles_in_component(self, component: Set[T]) -> int:
        """
        Count the number of independent cycles in a connected component.

        Uses the standard formula for an undirected connected graph:

            cycles = E - V + 1

        where:
        - E is the number of edges fully inside the component.
        - V is the number of nodes in the component.

        Notes
        -----
        - Returns 0 for a tree (no cycles).
        - Assumes the component is connected. `find_connected_components()`
          guarantees this for its outputs.
        """
        edge_count = sum(1 for a, b in self.edges if a in component and b in component)
        return edge_count - len(component) + 1

    def get_component_info(self) -> List[Dict[str, object]]:
        """
        Get information about all connected components.

        For each component, returns:
        - "nodes": set of nodes in the component
        - "cycles": number of cycles in that component

        """
        components = self.find_connected_components()
        return [
            {
                "nodes": comp,
                "cycles": self.count_cycles_in_component(comp),
            }
            for comp in components
        ]

    def find_components_of_size(self, size: int) -> List[Set[T]]:
        """
        Return all connected components whose number of nodes == `size`.
        """
        return [comp for comp in self.find_connected_components() if len(comp) == size]

    def find_triangles(self) -> List[Tuple[T, T, T]]:
        """
        Find all triangles (3-cliques) in the graph.

        A triangle is a triple of nodes (a, b, c) such that
        all three edges (a-b, b-c, a-c) exist.

        Returns a list of sorted tuples (a, b, c) with a < b < c
        to avoid duplicates.
        """
        triangles: List[Tuple[T, T, T]] = []
        # Sort nodes to enforce a < b < c ordering
        for a in sorted(self.nodes):
            neighbors_a = self.adjacency[a]
            # Only consider neighbors b > a to avoid duplicates
            for b in sorted(neighbors_a):
                if b <= a:
                    continue
                # Common neighbors of a and b are potential c's
                common = self.adjacency[a].intersection(self.adjacency[b])
                for c in common:
                    if c <= b:
                        continue  # enforce a < b < c
                    triangles.append((a, b, c))
        return triangles

    def largest_connected_component(self) -> Set[T]:
        """
        Return the connected component with the most nodes.
        If the graph is empty, return an empty set.
        """
        components = self.find_connected_components()
        return max(components, key=len) if components else set()

    def find_max_clique(self) -> Set[T]:
        """
        Find one maximum clique in the graph.

        A clique is a set of nodes where every pair is directly connected by an edge.
        This returns a single clique of maximum size (there may be others of same size).

        Note
        ----
        - Exponential in the worst case (NP-hard), but works well for modest graphs.
        """
        max_clique: Set[T] = set()
        adjacency = self.adjacency  # local alias for speed

        def bron_kerbosch(R: Set[T], P: Set[T], X: Set[T]) -> None:
            nonlocal max_clique

            # If no more candidates and no more excluded, R is a maximal clique
            if not P and not X:
                if len(R) > len(max_clique):
                    max_clique = set(R)
                return

            # Choose a pivot u from P ∪ X (heuristic to reduce branching)
            if P or X:
                u = max(P | X, key=lambda v: len(adjacency[v]))
            else:
                u = None

            # Explore candidates not connected to pivot u
            candidates = P - (adjacency[u] if u is not None else set())

            for v in list(candidates):
                bron_kerbosch(
                    R | {v},
                    P & adjacency[v],
                    X & adjacency[v],
                )
                P.remove(v)
                X.add(v)

        bron_kerbosch(set(), set(self.nodes), set())
        return max_clique

    def __repr__(self) -> str:
        """Compact string representation of the graph."""
        return f"Graph(nodes={self.nodes!r}, edges={self.edges!r})"


class DirectedGraph(Generic[T]):
    """
    Directed graph with support for one-way edges.

    Attributes
    ----------
    nodes : set[T]
        All nodes in the graph.
    edges : set[tuple[T, T]]
        Set of directed edges (a, b) meaning a -> b.
    successors : dict[T, set[T]]
        For each node, the set of nodes it points to (outgoing edges).
    predecessors : dict[T, set[T]]
        For each node, the set of nodes that point to it (incoming edges).

    Notes
    -----
    - No parallel edges are stored (edges are in a set).
    - The graph is directed: edge (a, b) means you can go from a to b,
      but NOT from b to a (unless (b, a) is also added).
    """

    def __init__(self) -> None:
        self.nodes: Set[T] = set()
        self.edges: Set[Tuple[T, T]] = set()
        self.successors: DefaultDict[T, Set[T]] = defaultdict(set)
        self.predecessors: DefaultDict[T, Set[T]] = defaultdict(set)

    def add_node(self, node: T) -> None:
        """
        Add a single node to the graph.
        """
        self.nodes.add(node)

    def add_edge(self, pair: Tuple[T, T]) -> None:
        """
        Add a directed edge from a to b.

        Both nodes are added if not already present.
        """
        a, b = pair
        self.add_node(a)
        self.add_node(b)

        self.edges.add((a, b))
        self.successors[a].add(b)
        self.predecessors[b].add(a)

    def has_edge(self, a: T, b: T) -> bool:
        """
        Check if there is a directed edge from a to b.
        """
        return (a, b) in self.edges

    def out_degree(self, node: T) -> int:
        """
        Return the number of outgoing edges from a node.
        """
        return len(self.successors[node])

    def in_degree(self, node: T) -> int:
        """
        Return the number of incoming edges to a node.
        """
        return len(self.predecessors[node])

    def get_successors(self, node: T) -> Set[T]:
        """
        Return the set of nodes that this node points to.
        """
        return self.successors[node]

    def get_predecessors(self, node: T) -> Set[T]:
        """
        Return the set of nodes that point to this node.
        """
        return self.predecessors[node]

    def is_reachable(self, start: T, end: T) -> bool:
        """
        Check if there is a directed path from start to end.

        Uses BFS to find if end is reachable from start.
        """
        if start not in self.nodes or end not in self.nodes:
            return False
        if start == end:
            return True

        visited: Set[T] = set()
        queue = [start]

        while queue:
            current = queue.pop(0)
            if current == end:
                return True
            if current in visited:
                continue
            visited.add(current)
            queue.extend(self.successors[current] - visited)

        return False

    def topological_sort(self) -> List[T]:
        """
        Return a topological ordering of the nodes.

        A topological ordering is a linear ordering of nodes such that
        for every directed edge (a, b), node a comes before node b.

        Returns
        -------
        list[T]
            Nodes in topological order.

        Raises
        ------
        ValueError
            If the graph contains a cycle (no topological order exists).
        """
        in_degree: Dict[T, int] = {node: self.in_degree(node) for node in self.nodes}
        queue = [node for node in self.nodes if in_degree[node] == 0]
        result: List[T] = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for successor in self.successors[node]:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        if len(result) != len(self.nodes):
            raise ValueError("Graph contains a cycle; topological sort not possible")

        return result

    def has_cycle(self) -> bool:
        """
        Check if the directed graph contains a cycle.
        """
        try:
            self.topological_sort()
            return False
        except ValueError:
            return True

    def find_all_paths(self, start: T, end: T) -> List[List[T]]:
        """
        Find all simple paths from start to end.

        A simple path visits each node at most once.

        Warning: This can be exponential in the number of paths!
        """
        if start not in self.nodes or end not in self.nodes:
            return []

        all_paths: List[List[T]] = []

        def dfs(current: T, path: List[T]) -> None:
            if current == end:
                all_paths.append(path[:])
                return
            for successor in self.successors[current]:
                if successor not in path:
                    path.append(successor)
                    dfs(successor, path)
                    path.pop()

        dfs(start, [start])
        return all_paths

    def reverse(self) -> "DirectedGraph[T]":
        """
        Return a new graph with all edges reversed.

        The edge (a, b) becomes (b, a).
        """
        reversed_graph: DirectedGraph[T] = DirectedGraph()
        for node in self.nodes:
            reversed_graph.add_node(node)
        for a, b in self.edges:
            reversed_graph.add_edge((b, a))
        return reversed_graph

    def strongly_connected_components(self) -> List[Set[T]]:
        """
        Find all strongly connected components using Kosaraju's algorithm.

        A strongly connected component is a maximal set of nodes where
        every node is reachable from every other node.
        """
        # First DFS to get finishing order
        visited: Set[T] = set()
        finish_order: List[T] = []

        def dfs1(node: T) -> None:
            visited.add(node)
            for successor in self.successors[node]:
                if successor not in visited:
                    dfs1(successor)
            finish_order.append(node)

        for node in self.nodes:
            if node not in visited:
                dfs1(node)

        # Second DFS on reversed graph in reverse finishing order
        reversed_graph = self.reverse()
        visited.clear()
        components: List[Set[T]] = []

        def dfs2(node: T, component: Set[T]) -> None:
            visited.add(node)
            component.add(node)
            for successor in reversed_graph.successors[node]:
                if successor not in visited:
                    dfs2(successor, component)

        for node in reversed(finish_order):
            if node not in visited:
                component: Set[T] = set()
                dfs2(node, component)
                components.append(component)

        return components

    def find_paths_through(self, start: T, end: T, must_visit: Set[T]) -> List[List[T]]:
        """
        Find all simple paths from start to end that visit ALL nodes in must_visit.

        Parameters
        ----------
        start : T
            Starting node.
        end : T
            Ending node.
        must_visit : set[T]
            Set of nodes that must appear in the path.

        Returns
        -------
        list[list[T]]
            All paths from start to end that pass through every node in must_visit.

        Notes
        -----
        - Prunes paths early if required nodes can no longer be visited.
        - Uses reachability checks to avoid exploring dead-end branches.
        """
        if start not in self.nodes or end not in self.nodes:
            return []

        # Precompute which nodes can reach each required node (and end)
        # by doing BFS backwards from each target
        targets = must_visit | {end}
        can_reach: Dict[T, Set[T]] = {}  # can_reach[target] = set of nodes that can reach target

        for target in targets:
            reachable_from: Set[T] = {target}
            queue = [target]
            while queue:
                node = queue.pop(0)
                for pred in self.predecessors[node]:
                    if pred not in reachable_from:
                        reachable_from.add(pred)
                        queue.append(pred)
            can_reach[target] = reachable_from

        # Check if start can reach all required nodes and end
        for target in targets:
            if start not in can_reach[target]:
                return []  # impossible

        all_paths: List[List[T]] = []

        # Use iterative DFS with explicit stack for better performance
        # Stack entries: (current_node, path_so_far, remaining_must_visit, successor_index)

        initial_remaining: frozenset[T] = frozenset(must_visit - {start})
        successors_list = {node: list(self.successors[node]) for node in self.nodes}

        stack: List[Tuple[T, List[T], frozenset[T], int]] = [(start, [start], initial_remaining, 0)]

        while stack:
            current, path, remaining, idx = stack.pop()

            if current == end:
                if not remaining:
                    all_paths.append(path[:])
                continue

            succs = successors_list[current]
            path_set = set(path)

            for i in range(idx, len(succs)):
                successor = succs[i]
                if successor in path_set:
                    continue

                new_remaining = remaining - {successor}

                # Pruning: can we still reach all remaining required nodes and end?
                reachable = True
                for target in new_remaining:
                    if successor not in can_reach[target]:
                        reachable = False
                        break
                if reachable and successor not in can_reach[end]:
                    reachable = False

                if reachable:
                    new_path = path + [successor]
                    stack.append((successor, new_path, new_remaining, 0))

        return all_paths

    def count_paths(self, start: T, end: T) -> int:
        """
        Count all simple paths from start to end.

        Uses memoization. Much faster than enumerating paths.
        Works well for DAGs; for cyclic graphs, still correct but slower.

        Parameters
        ----------
        start : T
            Starting node.
        end : T
            Ending node.

        Returns
        -------
        int
            Number of simple paths from start to end.
        """
        if start not in self.nodes or end not in self.nodes:
            return 0

        @lru_cache(maxsize=None)
        def dp(node: T, visited: frozenset[T]) -> int:
            if node == end:
                return 1

            total = 0
            for succ in self.successors[node]:
                if succ not in visited:
                    total += dp(succ, visited | {succ})
            return total

        return dp(start, frozenset({start}))

    def count_paths_dag(self, start: T, end: T) -> int:
        """
        Count all paths from start to end in a DAG.

        FAST: O(V + E) - no visited tracking needed for DAGs.

        Parameters
        ----------
        start : T
            Starting node.
        end : T
            Ending node.

        Returns
        -------
        int
            Number of paths from start to end.

        Note
        ----
        Only use this if the graph is a DAG (has_cycle() returns False).
        """
        if start not in self.nodes or end not in self.nodes:
            return 0

        @lru_cache(maxsize=None)
        def dp(node: T) -> int:
            if node == end:
                return 1
            return sum(dp(succ) for succ in self.successors[node])

        return dp(start)

    def count_paths_through_dag(self, start: T, end: T, must_visit: Set[T]) -> int:
        """
        Count paths from start to end that visit ALL nodes in must_visit (DAG version).

        FAST: O(V + E) per must_visit configuration.
        Only tracks which must_visit nodes have been seen, not all visited nodes.

        Parameters
        ----------
        start : T
            Starting node.
        end : T
            Ending node.
        must_visit : set[T]
            Set of nodes that must appear in the path.

        Returns
        -------
        int
            Number of valid paths.

        Note
        ----
        Only use this if the graph is a DAG (has_cycle() returns False).
        """
        if start not in self.nodes or end not in self.nodes:
            return 0

        if not must_visit:
            return self.count_paths_dag(start, end)

        must_list = list(must_visit)
        n = len(must_list)
        node_to_idx = {node: i for i, node in enumerate(must_list)}
        full_mask = (1 << n) - 1

        @lru_cache(maxsize=None)
        def dp(node: T, must_mask: int) -> int:
            """Count paths from node to end with must_mask tracking visited must_visit nodes."""
            if node == end:
                return 1 if must_mask == full_mask else 0

            total = 0
            for succ in self.successors[node]:
                new_mask = must_mask
                if succ in node_to_idx:
                    new_mask |= 1 << node_to_idx[succ]
                total += dp(succ, new_mask)
            return total

        initial_mask = 0
        if start in node_to_idx:
            initial_mask = 1 << node_to_idx[start]

        return dp(start, initial_mask)

    def count_paths_through(self, start: T, end: T, must_visit: Set[T]) -> int:
        """
        Count paths from start to end that visit ALL nodes in must_visit.

        Uses bitmask DP over the must_visit nodes. Much faster than enumeration.

        Parameters
        ----------
        start : T
            Starting node.
        end : T
            Ending node.
        must_visit : set[T]
            Set of nodes that must appear in the path (max ~15-20 for performance).

        Returns
        -------
        int
            Number of valid paths.
        """
        if start not in self.nodes or end not in self.nodes:
            return 0

        if not must_visit:
            return self.count_paths(start, end)

        must_list = list(must_visit)
        n = len(must_list)
        node_to_idx = {node: i for i, node in enumerate(must_list)}
        full_mask = (1 << n) - 1

        @lru_cache(maxsize=None)
        def dp(node: T, visited: frozenset[T], must_mask: int) -> int:
            """
            Count paths from node to end, having visited `visited` nodes,
            with must_mask indicating which must_visit nodes still need visiting.
            """
            if node == end:
                return 1 if must_mask == full_mask else 0

            total = 0
            for succ in self.successors[node]:
                if succ not in visited:
                    new_mask = must_mask
                    if succ in node_to_idx:
                        new_mask |= 1 << node_to_idx[succ]
                    total += dp(succ, visited | {succ}, new_mask)
            return total

        initial_mask = 0
        if start in node_to_idx:
            initial_mask = 1 << node_to_idx[start]

        return dp(start, frozenset({start}), initial_mask)

    def __repr__(self) -> str:
        """Compact string representation of the directed graph."""
        return f"DirectedGraph(nodes={self.nodes!r}, edges={self.edges!r})"
