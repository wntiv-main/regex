import traceback
from regex import *

if __name__ == "__main__":
    print("WARNING: Did you mean to run this module directly?")
    from debug_graph_viewer import MultiFigureViewer, DebugGraphViewer, \
        test_layouts_for


def main():
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
        fig.canvas.manager.set_window_title(msg)
        mfv.add(fig)
        print(f"{msg}:\n{rx}")
    Regex._debug_function = debug
    try:
        rx = Regex(r"\sa|.b")
        print(rx)
        mfv.add(DebugGraphViewer(rx.edge_map,
                                 rx.start, rx.end).render())
    except Exception:
        traceback.print_exc()
    finally:
        mfv.display()


if __name__ == "__main__":
    main()
