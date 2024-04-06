from regex_builder import *

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

    def debug(s, e, lbl):
        # pass
        fig = DebugGraphViewer(s, e).render()
        fig.canvas.manager.set_window_title(lbl)
        mfv.add(fig)
    try:
        rx = RegexBuilder(r"(?:a?b?c?)*").build(
            debug=debug)
        while i := input():
            print(i in rx)
        mfv.add(DebugGraphViewer(rx.start, rx.end).render())
    except KeyboardInterrupt as e:
        mfv.display()
        raise e
    mfv.display()
    #     print(e)
    #     raise e
    # finally:
    # test_layouts_for(rx.begin(), rx.end())

if __name__ == "__main__":
    main()
