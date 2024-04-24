"""Example showing the debug graph viewer"""

from regex import Regex
from regex.debug_graph_viewer import MultiFigureViewer, DebugGraphViewer


def main():
    """Entrypoint"""
    mfv = MultiFigureViewer()

    def on_optimisation_step(rx: Regex, msg: str):
        # Create a figure showing what was done this step
        view = DebugGraphViewer(rx.edge_map,
                                rx.start, rx.end)
        fig = view.render()
        fig.suptitle(str(rx), fontsize=8)
        fig.canvas.manager.set_window_title(msg)  # type: ignore
        mfv.add(fig)

    # Hook into regex's internal logging
    # pylint: disable-next=protected-access
    Regex._debug_function = on_optimisation_step

    # Email regex
    rx = Regex(r"\A\w+(?:\.\w+)*@\w+(?:\.\w+)+\Z")
    # Add final result
    mfv.add(DebugGraphViewer(rx.edge_map, rx.start, rx.end).render())
    mfv.display()


if __name__ == "__main__":
    main()
