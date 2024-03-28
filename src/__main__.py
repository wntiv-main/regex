from regex import *

if __name__ == "__main__":
    print("WARNING: Did you mean to run this module directly?")
    from debug_graph_viewer import MultiFigureViewer, DebugGraphViewer


def main():
    # r"(fn|(\w+)) \(((\w+)\s*(\w+))*\)\s*(=>\s*(\w+))?"
    # r"^(?P<user>(?:\w|\.|\+|-)+)@(?P<domain>(?:\w+\.)+\w+)$"
    # rx = RegexBuilder(
    #     r"(?:def|function|(?P<rtype>\w+))\s*(?P<fname>\w+)\s*" \
    #     r"\((?:(?P<argtype>\w+)\s*(?P<argname>\w+))*\)\s*" \
    #     r"(?:=>\s*(?P<rtype>\w+))?").build()
    mfv = MultiFigureViewer()

    def debug(s, e, lbl):
        fig = DebugGraphViewer(s, e).render()
        fig.canvas.manager.set_window_title(lbl)
        mfv.add(fig)
    # try:
    rx = RegexBuilder(r"(?:a?b?c?)*").build(
        debug=debug)
    mfv.add(DebugGraphViewer(rx.begin(), rx.end()).render())
    # finally:
    mfv.display()
    # test_layouts_for(rx.begin(), rx.end())

if __name__ == "__main__":
    main()
