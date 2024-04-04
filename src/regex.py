from enum import IntFlag, auto
from typing import Callable


from funcutil import wrap_method
from regexutil import Direction, State, Edge, MatchConditions


class Regex:
    _start: State
    _end: State
    _states: set[State] | None = None

    @property
    def start(self) -> State:
        while self._start._replaced_with is not None:
            self._start = self._start._replaced_with
        return self._start

    @start.setter
    def start(self, value: State) -> None:
        self._start = value

    @property
    def end(self) -> State:
        while self._end._replaced_with is not None:
            self._end = self._end._replaced_with
        return self._end

    @end.setter
    def end(self, value: State) -> None:
        self._end = value

    def states(self):
        if self._states is None:
            self._states = set()
            to_explore: list[State] = [self._start]
            while to_explore:
                exploring = to_explore.pop()
                if exploring in self._states:
                    continue
                self._states.add(exploring)
                for edge in exploring.next:
                    if edge.next == exploring:
                        continue
                    to_explore.append(edge.next)
        return self._states

    # _flags: RegexFlags
    # _capture_groups: set[CaptureGroup]

    def __init__(self, start, end) -> None:
        self._start = start
        self._end = end

    class RecursionType(IntFlag):
        FORWARD = auto()
        REVERSE = auto()
        BOTH = FORWARD | REVERSE

    def walk_graph(
            self,
            visitor: Callable[[Edge], State | None],
            *args, _start: State | None = None,
            _visited: set[State] | None = None,
            _side: RecursionType = RecursionType.FORWARD,
            **kwargs) -> Callable[['Regex'], None]:
        if _start is None:
            _start = self._start
        while _start._replaced_with is not None:
            _start = _start._replaced_with
        if _visited is None:
            _visited = set()
        if _start in _visited:
            return
        _visited.add(_start)
        for edge in _start.next.copy():
            if edge.next is None or edge.previous is None:
                continue
            if _side & Regex.RecursionType.FORWARD:
                visitor(
                    edge, *args, **kwargs,
                    _start=self._start, _end=self._end)
            if edge.next is not None:
                self.walk_graph(visitor, *args,
                                **kwargs,
                                _start=edge.next,
                                _visited=_visited)
            else:
                _visited.discard(_start)
                self.walk_graph(visitor, *args,
                                **kwargs,
                                _start=_start,
                                _visited=_visited)
                return
            if _side & Regex.RecursionType.REVERSE:
                visitor(
                    edge, *args, **kwargs,
                    _start=self._start, _end=self._end)

    @wrap_method(walk_graph)
    def epsilon_closure(
            edge: Edge,
            _start: State,
            _end: State,
            debug) -> None:
        if (edge.is_free() and edge.previous == edge.next):
            next_state = edge.previous
            with edge:
                edge.remove()
            debug(_start, _end, f"remove self-loop from {next_state}")
            return
        # transfer capture groups to non-epsilon transitions
        if (edge.predicate == MatchConditions.epsilon_transition
                and (edge.has_opens() or edge.has_closes())):
            with edge:
                if edge.has_opens() and edge.next.inputs() == 1:
                    try:  # spoof with statement dont question it
                        for path in edge.next.next:
                            path.__enter__()
                        for group in edge.move_opens():
                            for path in edge.next.next:
                                path.open(group)
                    finally:
                        for path in edge.next.next:
                            path.__exit__(*(None,)*3)
                if edge.has_closes() and edge.previous.outputs() == 1:
                    try:  # spoof with statement dont question it
                        for path in edge.previous.previous:
                            path.__enter__()
                        for group in edge.move_closes():
                            for path in edge.previous.previous:
                                path.close(group)
                    finally:
                        for path in edge.previous.previous:
                            path.__exit__(*(None,)*3)
        # merge states connected by e-moves
        if (edge.is_free()
            and (edge.previous.outputs() == 1
                 or edge.next.inputs() == 1)):
            debug_str = f"{edge}: merge {edge.next} with {edge.previous}"
            edge.next.merge(edge.previous)
            debug(_start, _end, debug_str)
            return

    @wrap_method(walk_graph, _side=RecursionType.REVERSE)
    def extended_epsilon_closure(
            edge: Edge,
            _start: State,
            _end: State,
            debug) -> None:
        if (edge.is_free() and edge.previous == edge.next):
            next_state = edge.previous
            with edge:
                edge.remove()
            debug(_start, _end, f"remove self-loop from {next_state}")
            return
        # strategy for removing enclosed e-moves: split their end-state
        # into 2 - one for the e-move, one for the other connections
        if (edge.is_free()
                and edge.previous.outputs() > 1
                and edge.next.inputs() > 1):
            debug(_start, _end, f"{edge}: spliting {edge.next}")
            new_state = edge.next.clone_shallow(Direction.FORWARD)
            edge.previous.merge(new_state)
            if edge.next == _end:
                with Edge() as new_edge:
                    new_edge.previous = edge.previous
                    new_edge.next = _end
            with edge:
                edge.remove()
            debug(_start, _end, f"{edge}: split {edge.next} to {new_state}")

    @wrap_method(walk_graph)
    def powerset_construction(
            edge: Edge,
            _start: State,
            _end: State,
            debug) -> None:
        for other in edge.previous.next.copy():
            if edge.next == other.next:
                continue
            match edge.predicate_intersection(other):
                case None: continue
                case left, intersect, right:
                    # We need to construct a "superposition" state
                    start_state = edge.previous
                    debug(_start, _end,
                          f"intersect {edge}, {other} at {start_state}")
                    new_state = edge.next.clone_shallow(Direction.FORWARD)
                    new_state.merge(
                        other.next.clone_shallow(Direction.FORWARD))
                    with other:
                        if right is not None:
                            other.predicate = right
                        else:
                            other.remove(chain=True)
                    with Edge(intersect) as intersect_edge:
                        intersect_edge.previous = start_state
                        intersect_edge.next = new_state
                    with edge:
                        if left is not None:
                            edge.predicate = left
                        else:
                            edge.remove(chain=True)
                            break  # edge removed, no more loop


    def minify(self):
        states: set[State] = set()
        to_explore: list[State] = [self._start]
        while to_explore:
            exploring = to_explore.pop()
            if exploring in states:
                continue
            states.add(exploring)
            for edge in exploring.next:
                if edge.next == exploring:
                    continue
                to_explore.append(edge.next)
        states_list = list(states)
        for i in range(len(states_list) - 1):  # exclude last
            first = states_list[i]
            # only iterate forward states
            for second in states_list[i + 1:]:
                diff = first.output_diff(second)
                for edge in diff:
                    # TODO: this is wrong. fix
                    if not edge.is_free() or (edge.next != first
                                              and edge.next != second):
                        break
                else:
                    first.merge(second)

    def match_in(self, string: str) -> bool:
        ctx = MatchConditions(string)
        current_state = self._start
        while current_state != self._end:
            for edge in current_state.next:
                if edge.predicate(ctx):
                    current_state = edge.next
                    break
            else:
                return False
        return True
    __contains__ = match_in
