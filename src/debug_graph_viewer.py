import math
from typing import Callable, TypeVar

# REFERENCE: https://www.geeksforgeeks.org/visualize-graphs-in-python/

try:
    import numpy as np
    import networkx
    from networkx import layout as nxlayout
    import matplotlib.pyplot
    import matplotlib.figure
    from matplotlib.widgets import Button
except ImportError:
    print("If you wish to proceed to display the debug graphic, "
          "`$ pip install networkx`, `$ pip install matplotlib`, "
          "`$ pip install pyqt5`, and `$ pip install scipy`")
    quit()
import networkx_curved_label

T = TypeVar('T')


class DebugGraphViewer:
    _graph: networkx.MultiDiGraph
    _layout: dict[tuple[int, int], tuple[float, float]] | None
    _layout_planner: Callable
    _color_overrides: dict[int, tuple[float, float, float]]

    def __init__(
            self,
            graph: np.ndarray[set[T]],
            start_idx: int,
            end_idx: int,
            layout=networkx.layout.kamada_kawai_layout):
        self._graph = networkx.MultiDiGraph()
        self._layout = None
        self._layout_planner = layout
        self._color_overrides = {}
        self._color_overrides[start_idx] = (1.0, 0.3, 0.3)
        self._color_overrides[end_idx] = (0.3, 1.0, 0.3)
        # iterate graph
        it = np.nditer(graph, flags=['multi_index', 'refs_ok'])
        # if only python had a do: ... while() loop :(
        for edges in it:
            start_state, end_state = it.multi_index
            # add nodes
            self._graph.add_node(start_state, label=str(start_state))
            self._graph.add_node(end_state, label=str(end_state))
            # add edge
            if isinstance(edges[()], set):
                for edge in edges[()]:
                    self._graph.add_edge(start_state, end_state,
                                         label=str(edge))

    def _display_edges(self,
                       edges: list[tuple[int, int, int, str]],
                       rad: float = 0):
        edge_list = [(x, y, key) for x, y, key, label in edges]
        networkx.draw_networkx_edges(
            self._graph,
            self._layout,
            edgelist=edge_list,
            connectionstyle="arc3" if rad == 0 else f"arc3, rad = {rad}",
            node_size=300 // math.sqrt(self._graph.number_of_nodes()))
        labels = {(x, y, key): label for x, y, key, label in edges}
        networkx_curved_label.draw_networkx_edge_labels(
            self._graph,
            self._layout,
            labels,
            edgelist=edge_list,
            node_size=300 // math.sqrt(self._graph.number_of_nodes()),
            rad=rad,
            font_size=30 // (math.sqrt(self._graph.number_of_nodes())))

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
            node_size=100 // math.sqrt(self._graph.number_of_nodes()))
        networkx.draw_networkx_labels(
            self._graph,
            self._layout,
            labels={x: lbl for x, lbl in self._graph.nodes(data='label')},
            font_size=15 // math.sqrt(self._graph.number_of_nodes()))
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
        if len(self._figures) < 1:
            return
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


def test_layouts_for(graph: np.ndarray[set[T]],
                     start_idx: int,
                     end_idx: int):
    view = MultiFigureViewer()
    for name in nxlayout.__all__:
        layout = getattr(nxlayout, name)
        viewer = DebugGraphViewer(graph, start_idx, end_idx, layout)
        try:
            fig = viewer.render()
        except Exception:
            continue
        fig.canvas.manager.set_window_title(name)
        view.add(fig)
    view.display()
