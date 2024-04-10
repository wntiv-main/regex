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
        view = DebugGraphViewer(rx.transition_table,
                                rx.start, rx.end)
        fig = view.render()
        fig.suptitle(str(rx), fontsize=8)
        fig.canvas.manager.set_window_title(msg)
        mfv.add(fig)
    Regex._debug_function = debug

    rx = Regex(r"(?:a?b?c?)*")
    print(rx)
    mfv.add(DebugGraphViewer(rx.transition_table,
                             rx.start, rx.end).render())
    mfv.display()


if __name__ == "__main__":
    main()
