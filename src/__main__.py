from regex import *

if __name__ == "__main__":
    print("WARNING: This package is made to be imported not run directly")
    from debug_graph_viewer import *


def main():
    # r"(fn|(\w+)) \(((\w+)\s*(\w+))*\)\s*(=>\s*(\w+))?"
    rx = RegexBuilder(
        r"^(?P<user>(?:\w|\.|\+|-)+)@(?P<domain>(?:\w+\.)+\w+)$")\
        .build()
    test_layouts_for(rx.begin(), rx.end())

if __name__ == "__main__":
    main()
