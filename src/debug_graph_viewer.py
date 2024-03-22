from typing import Generic, TypeVar

# REFERENCE: https://www.geeksforgeeks.org/visualize-graphs-in-python/

try:
    import networkx
    import matplotlib.pyplot
except ImportError:
    print("If you wish to proceed to display the debug graphic, "
          "`$ pip install networkx` and `$ pip install matplotlib`")
import networkx_curved_label

N = TypeVar("N")  # Node
E = TypeVar("E")  # Edge


class DebugGraphViewer(Generic[N, E]):
    _auto_increment_id: int
    _visited_nodes: dict[N, int]
    _visited_edges: set[E]
    _graph: networkx.MultiDiGraph
    _layout: dict[tuple[int, int], tuple[float, float]] | None

    def __init__(self, start: N) -> None:
        self._auto_increment_id = 0
        self._visited_nodes = {}
        self._visited_edges = set()
        self._graph = networkx.MultiDiGraph()
        self.explore_node(start)

    def explore_node(self, node: N) -> int:
        if node in self._visited_nodes:
            return self._visited_nodes[node]
        print(f"at {node=}")
        id = self._auto_increment_id
        self._auto_increment_id += 1
        self._visited_nodes[node] = id
        self._graph.add_node(id)
        for edge in node.next:
            self.explore_edge(edge)
        for edge in node.previous:
            self.explore_edge(edge)
        return id

    def explore_edge(self, edge: E) -> None:
        if edge in self._visited_edges:
            return
        print(f"at {edge=}")
        self._visited_edges.add(edge)
        start = self.explore_node(edge.previous)
        end = self.explore_node(edge.next)
        self._graph.add_edge(start, end, label=repr(edge))

    def _display_edges(self,
                       edges: list[tuple[int, int, int, str]],
                       rad: float = 0):
        edge_list = [(x, y, key) for x, y, key, label in edges]
        networkx.draw_networkx_edges(
            self._graph,
            self._layout,
            edgelist=edge_list,
            connectionstyle="arc3" if rad == 0 else f"arc3, rad = {rad}")
        labels = {(x, y, key): label for x, y, key, label in edges}
        networkx_curved_label.draw_networkx_edge_labels(
            self._graph,
            self._layout,
            labels,
            rad=rad)

    def display(self):
        # We need to display graph in multiple batches. This allows us
        # to draw multi- and directional- connections without overlap.
        # Inspired by: https://stackoverflow.com/a/70245742
        self._layout = networkx.layout.kamada_kawai_layout(self._graph)
        networkx.draw_networkx_nodes(
            self._graph,
            self._layout)
        # A list of edges for each connection between nodes
        edges_by_connection: dict[tuple[int, int],
                                  list[tuple[int, int, int]]] = {}
        for start, end, key, label in self._graph.edges(keys=True, data='label'):
            ordered = (min(start, end), max(start, end))
            if ordered in edges_by_connection:
                edges_by_connection[ordered].append((start, end, key, label))
            else:
                edges_by_connection[ordered] = [(start, end, key, label)]

        # exclude connections with multiple edges, and self-loops
        simple_edges = [i[0] for i in edges_by_connection.values()
                        if len(i) == 1 and i[0][0] != i[0][1]]

        self_loop_edges = [i for (x, y), l in edges_by_connection.items()
                           for i in l  # Flatten list
                           if x == y]
        self._display_edges(self_loop_edges, rad=1)

        complex_edges = [i for i in edges_by_connection.values()
                         if len(i) > 1 and i[0][0] != i[0][1]]
        rad = 0.25
        while complex_edges:
            edges_at_lvl = []
            alt_edges_at_lvl = []
            for i in complex_edges:
                if len(i) > 1:
                    # Pop and place first
                    edge = i.pop(0)
                    if edge[0] < edge[1]:
                        edges_at_lvl.append(edge)
                    else:
                        alt_edges_at_lvl.append(edge)
                    # Pop and place second on opposite side
                    edge = i.pop(0)
                    if edge[0] < edge[1]:
                        alt_edges_at_lvl.append(edge)
                    else:
                        edges_at_lvl.append(edge)
                else:
                    simple_edges.append(i.pop(0))
            # draw
            self._display_edges(edges_at_lvl, rad)
            self._display_edges(alt_edges_at_lvl, -rad)
            rad *= 2
            # Remove empty lists
            complex_edges = [i for i in complex_edges if i]

        self._display_edges(simple_edges)

        # curved_edges = [
        #     edge for edge in G.edges() if reversed(edge) in G.edges()]
        # straight_edges = list(set(G.edges()) - set(curved_edges))
        # nx.draw_networkx_edges(G, pos, ax=ax, edgelist=straight_edges)
        # arc_rad = 0.25
        # nx.draw_networkx_edges(
        #     G, pos, ax=ax, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}')

        # networkx.draw_networkx(
        #     self._graph,
        #     layout,
        #     connectionstyle='arc3, rad = 0.1')
        # networkx.draw_networkx_edge_labels(
        #     self._graph,
        #     layout,
        #     self._labels)
        matplotlib.pyplot.show()


if __name__ == "__main__":
    class TestEdge:
        name: str
        previous: 'TestNode'
        next: 'TestNode'

        def __init__(self,
                     prev: 'TestNode',
                     next: 'TestNode',
                     name: str):
            self.previous = prev
            self.next = next
            self.name = name

        def __repr__(self) -> str:
            return self.name

    class TestNode:
        previous: set[TestEdge]
        next: set[TestEdge]

        def __init__(self):
            self.previous = set()
            self.next = set()

        def connect(self, other: 'TestNode', label: str) -> 'TestEdge':
            e = TestEdge(self, other, label)
            self.next.add(e)
            other.previous.add(e)

    n1 = TestNode()
    n2 = TestNode()
    n3 = TestNode()
    n4 = TestNode()
    n5 = TestNode()
    n1.connect(n2, "path1")
    n2.connect(n2, "loop")
    n2.connect(n2, "altloop")
    n2.connect(n3, "continue")
    n1.connect(n4, "path2")
    n1.connect(n4, "path3")
    n4.connect(n3, "continue")
    n4.connect(n3, "continuealt")
    n3.connect(n4, "reverse")
    n3.connect(n5, "end")
    n4.connect(n5, "end")
    DebugGraphViewer(n1).display()
