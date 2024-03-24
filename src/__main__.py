from regex import *
from debug_graph_viewer import DebugGraphViewer, MultiFigureViewer, nxlayout

if __name__ == "__main__":
    print("WARNING: This package is made to be imported not run directly")
    import debug_graph_viewer


def main():
    # r"(fn|(\w+)) \(((\w+)\s*(\w+))*\)\s*(=>\s*(\w+))?"
    mfv = MultiFigureViewer()
    rx = RegexBuilder(
        r"(?:fn|(?P<rtype>\w+))\s*\((?:(?P<argtype>\w+)\s*(?P<argname>\w+))*\)\s*(?:=>\s*(?<rtype>\w+))?")\
        .build(debug=lambda x:
               mfv.add(DebugGraphViewer(x).render()) if x.next else None)
    mfv.add(DebugGraphViewer(
        rx.begin(), rx.end(),
        layout=nxlayout.kamada_kawai_layout).render())
    mfv.display()


if __name__ == "__main__":
    main()
