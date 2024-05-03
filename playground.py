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
    #     r"\((?:(?P<argtype>\w+)\s*(?P<argname>\w+))*\)\s*" \sillly
    #     r"(?:=>\s*(?P<rtype>\w+))?").build()
    mfv = MultiFigureViewer()

    def debug(rx: Regex, msg: str):
        view = DebugGraphViewer(rx.edge_map,
                                rx.start, rx.end)
        fig = view.render()
        fig.suptitle(str(rx), fontsize=12 - rx.size // 5)
        fig.canvas.manager.set_window_title(msg)  # type: ignore
        mfv.add(fig)
        print(f"{msg}:\n{rx}")
        # print(f"dupls: {rx._find_double_refs()}")
    # Regex._debug_function = debug
    try:
        # r"\w+(?:\.\w+)*@\w+(?:\.\w+)+"
        # r"(?:ca{1,2}b*)?a"
        # r"(?:\+\d{1,2}\s*)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}"
        for i in range(10):
            rx = Regex(r"(?P<user>\w+(?:\.\w+)*)@(?P<domain>\w+(?:\.\w+)+)")
            print(f"Done {i}")
        assert rx._base is not None
        # while (i := input("email??: ")):
        #     print(rx.test(i))
        mfv.add(DebugGraphViewer(rx.edge_map,
                                 rx.start, rx.end).render())
        # rxr = rx.reverse()
        # # while (i := input("emailish??: ")):
        # #     print(rx.replace_in(i, '(%0)'))
        # mfv.add(DebugGraphViewer(rxr.edge_map,
        #                          rxr.start, rxr.end).render())
    except Exception:
        # Print exception, still show viewer
        traceback.print_exc()
    finally:
        mfv.display()


if __name__ == "__main__":
    main()
