"""Playground for debug testing"""

import traceback
# In production, this would be a seperate package, and would thus be
# imported with something akin to `from regex import Regex`
from regex import Regex
from regex.debug_graph_viewer import MultiFigureViewer, DebugGraphViewer


def main():
    """Main function, called on execution"""
    # r"(fn|(\w+)) \(((\w+)\s*(\w+))*\)\s*(=>\s*(\w+))?"
    # r"^(?P<user>(?:\w|\.|\+|-)+)@(?P<domain>(?:\w+\.)+\w+)$"
    # rx = RegexBuilder(
    #     r"(?:def|function|(?P<rtype>\w+))\s*(?P<fname>\w+)\s*" \
    #     r"\((?:(?P<argtype>\w+)\s*(?P<argname>\w+))*\)\s*" \
    #     r"(?:=>\s*(?P<rtype>\w+))?").build()
    mfv = MultiFigureViewer()

    def debug(rx: Regex, msg: str):
        view = DebugGraphViewer(rx.edge_map,
                                rx.start, rx.end)
        fig = view.render()
        fig.suptitle(str(rx), fontsize=8)
        fig.canvas.manager.set_window_title(msg)  # type: ignore
        mfv.add(fig)
        print(f"{msg}:\n{rx}")
        print(f"dupls: {rx._find_double_refs()}")
    Regex._debug_function = debug
    try:
        rx = Regex(r"a*b*c*")
        print(rx)
        mfv.add(DebugGraphViewer(rx.edge_map,
                                 rx.start, rx.end).render())
    except Exception:
        # Print exception, still show viewer
        traceback.print_exc()
    finally:
        mfv.display()


if __name__ == "__main__":
    main()
