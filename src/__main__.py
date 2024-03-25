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

    rx = RegexBuilder(r"(?:a?b?c?)*").build(
        debug=lambda s: mfv.add(DebugGraphViewer(s).render()))
    mfv.add(DebugGraphViewer(rx.begin(), rx.begin()).render())
    mfv.display()
    # test_layouts_for(rx.begin(), rx.end())

if __name__ == "__main__":
    main()
