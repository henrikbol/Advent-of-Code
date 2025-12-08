from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable

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

    def __repr__(self) -> str:
        """Compact string representation of the graph."""
        return f"Graph(nodes={self.nodes!r}, edges={self.edges!r})"
