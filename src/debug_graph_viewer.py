import math
from typing import Callable, Generic, TypeVar

# REFERENCE: https://www.geeksforgeeks.org/visualize-graphs-in-python/

try:
    import networkx
    from networkx import layout as nxlayout
    import matplotlib.pyplot
    import matplotlib.figure
    from matplotlib.widgets import Button
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
    _layout_planner: Callable
    _color_overrides: dict[int, tuple[float, float, float]]

    def __init__(
            self,
            start: N,
            end: N | None = None,
            layout=networkx.layout.kamada_kawai_layout):
        self._auto_increment_id = 0
        self._visited_nodes = {}
        self._visited_edges = set()
        self._graph = networkx.MultiDiGraph()
        self._layout = None
        self._layout_planner = layout
        self._color_overrides = {
            self.explore_node(start): (1.0, 0.3, 0.3)
        }
        if end is not None:
            self._color_overrides[self.explore_node(end)] = (0.3, 1.0, 0.3)

    def explore_node(self, node: N, color=None) -> int:
        if node in self._visited_nodes:
            return self._visited_nodes[node]
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
            connectionstyle="arc3" if rad == 0 else f"arc3, rad = {rad}",
            node_size=300 // math.log(self._graph.number_of_nodes()))
        labels = {(x, y, key): label for x, y, key, label in edges}
        networkx_curved_label.draw_networkx_edge_labels(
            self._graph,
            self._layout,
            labels,
            rad=rad,
            font_size=30 // (math.log(self._graph.number_of_nodes())))

    def render(self) -> matplotlib.figure.Figure:
        fig = matplotlib.pyplot.figure(layout='tight')
        # We need to display graph in multiple batches. This allows us
        # to draw multi- and directional- connections without overlap.
        # Inspired by: https://stackoverflow.com/a/70245742
        self._layout = self._layout_planner(self._graph)
        colors = {
            node: self._color_overrides[node]
            if node in self._color_overrides
            else (0.3, 0.3, 1.0)
            for node in self._graph.nodes
        }
        networkx.draw_networkx_nodes(
            self._graph,
            self._layout,
            nodelist=list(colors.keys()),
            node_color=list(colors.values()),
            node_size=100 // math.log(self._graph.number_of_nodes()))
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
        rad = 0.5  # * math.log(self._graph.number_of_nodes())
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

        return fig

    @staticmethod
    def display():
        matplotlib.pyplot.show()


class MultiFigureViewer:
    _current: int
    _figures: list[matplotlib.figure.Figure]
    _visited: set[matplotlib.figure.Figure]
    _buttons: dict[matplotlib.figure.Figure, tuple[Button, Button]]
    _last_fig: matplotlib.figure.Figure
    _btn_fig: matplotlib.figure.Figure

    def __init__(self) -> None:
        self._figures = []
        self._visited = set()
        self._buttons = {}
        self._current = 0
        self._last_fig = None

    def add(self, fig: matplotlib.figure.Figure):
        fig.subplots_adjust(bottom=0.2)

        axprev = fig.add_axes([0.7, 0.05, 0.1, 0.075])
        axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])

        btn_next = Button(axnext, 'Next')
        btn_next.on_clicked(self.next)
        btn_prev = Button(axprev, 'Previous')
        btn_prev.on_clicked(self.prev)
        self._buttons[fig] = btn_next, btn_prev

        # Maximize window:
        # https://stackoverflow.com/questions/12439588/how-to-maximize-a-plt-show-window
        match matplotlib.get_backend():
            # fig.canvas.manager.window.maximize()
            case 'Qt4Agg' | 'QtAgg':
                pass  # Handled later
            case  'wxAgg':
                fig.canvas.manager.frame.Maximize(True)
            case 'TkAgg':
                try:
                    fig.canvas.manager.window.state('zoomed')
                except Exception:
                    fig.canvas.manager.resize(
                        *fig.canvas.manager.window.maxsize())

        self._figures.append(fig)

    def next(self, e):
        self._current += 1
        if self._current >= len(self._figures):
            self._current = 0
        self._display()

    def prev(self, e):
        self._current -= 1
        if self._current < 0:
            self._current = len(self._figures) - 1
        self._display()

    def _display(self):
        if self._last_fig is not None:
            # Hacks, just hope this works
            self._last_fig.canvas.manager.window.close()
        self._last_fig = self._figures[self._current]
        self._last_fig.show()
        if (self._last_fig not in self._visited
                and matplotlib.get_backend() == 'QtAgg'):
            self._last_fig.canvas.manager.window.showMaximized()
        self._visited.add(self._last_fig)

    def display(self):
        self._current = len(self._figures) - 1
        self._display()
        matplotlib.pyplot.get_current_fig_manager().start_main_loop()


def test_layouts_for(start: N, end: N | None = None):
    view = MultiFigureViewer()
    for name in networkx.layout.__all__:
        try:
            layout = getattr(networkx.layout, name)
            fig = DebugGraphViewer(start, end, layout).render()
            fig.canvas.manager.set_window_title(name)
        except Exception:
            continue
        view.add(fig)
    view.display()

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
    DebugGraphViewer(n1).render()
    DebugGraphViewer.display()
